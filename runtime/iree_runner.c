// SPDX-License-Identifier: Apache-2.0
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "iree/runtime/api.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/hal/local/loaders/static_library_loader.h"
#include "iree/vm/bytecode/module.h"
#include "libraries.h"
#include "proton/platform.h"
#ifdef PROTON_MODEL
#include <math.h>
#include "model_state.h"
#include "linear.h"
static proton_state_t state;
#ifndef PROTON_STEPS
#define PROTON_STEPS 16
#endif
#ifndef PROTON_BACKEND
#define PROTON_BACKEND 1
#endif
static uint64_t cycles(void) { uint64_t c; __asm__ volatile("rdcycle %0":"=r"(c)); return c; }
static unsigned bits(float f) { unsigned b; memcpy(&b,&f,4); return b; }
#endif
extern const unsigned char module_data[];
extern const size_t module_size;
extern uintptr_t proton_heap_peak;

#define CHECK(expr) do { iree_status_t s_ = (expr); if (!iree_status_is_ok(s_)) { \
  printf("IREE FAIL line=%d status=%d\n", __LINE__, (int)iree_status_code(s_)); \
  iree_status_ignore(s_); return 1; } } while(0)

int main(void) {
  printf("PROTON IREE START\n");
  if(proton_memory_begin()) { printf("FAIL memory initialization\n"); return 1; }
  iree_allocator_t alloc = iree_allocator_system();
  iree_runtime_instance_options_t io;
  iree_runtime_instance_options_initialize(&io);
  iree_runtime_instance_t *instance = NULL;
  CHECK(iree_runtime_instance_create(&io, alloc, &instance));
  iree_hal_executable_loader_t *loader = NULL;
  CHECK(iree_hal_static_library_loader_create(IREE_ARRAYSIZE(proton_libraries),
    proton_libraries, iree_hal_executable_import_provider_null(), alloc, &loader));
  iree_hal_allocator_t *device_alloc = NULL;
  CHECK(iree_hal_allocator_create_heap(iree_make_cstring_view("proton"), alloc, alloc, &device_alloc));
  iree_hal_sync_device_params_t params;
  iree_hal_sync_device_params_initialize(&params);
  iree_hal_device_t *device = NULL;
  CHECK(iree_hal_sync_device_create(iree_make_cstring_view("local-sync"), &params,
    1, &loader, device_alloc, alloc, &device));
  iree_hal_executable_loader_release(loader);
  iree_hal_allocator_release(device_alloc);
  iree_runtime_session_options_t so;
  iree_runtime_session_options_initialize(&so);
  iree_runtime_session_t *session = NULL;
  CHECK(iree_runtime_session_create_with_device(instance, &so, device, alloc, &session));
  iree_vm_module_t *module = NULL;
  CHECK(iree_vm_bytecode_module_create(iree_runtime_instance_vm_instance(instance), 0,
    iree_make_const_byte_span(module_data, module_size), iree_allocator_null(), alloc, &module));
  CHECK(iree_runtime_session_append_module(session, module));
  iree_runtime_call_t call;
#ifdef PROTON_MODEL
  if(!proton_matrix_probe()) { printf("FAIL matrix identity\n"); return 2; }
  uint32_t zero_start=proton_matrix_count(0x10),macc_start=proton_matrix_count(0x14);
  state.token=1; state.backend=PROTON_BACKEND;
  uint64_t model_cycles=0; float max_error=0;
  for(unsigned pos=0;pos<PROTON_STEPS;++pos) {
    state.position=(float)pos;
    CHECK(iree_runtime_call_initialize_by_name(session, iree_make_cstring_view("proton_model.step"), &call));
    iree_hal_dim_t shape[]={sizeof(state)/sizeof(float)};
    iree_hal_buffer_view_t *input=NULL;
    CHECK(iree_hal_buffer_view_allocate_buffer_copy(device,iree_hal_device_allocator(device),1,shape,
      IREE_HAL_ELEMENT_TYPE_FLOAT_32,IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){.type=IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,.usage=IREE_HAL_BUFFER_USAGE_DEFAULT},
      iree_make_const_byte_span(&state,sizeof(state)),&input));
    CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call,input));
    iree_hal_buffer_view_release(input);
    uint64_t begin=cycles();
    CHECK(iree_runtime_call_invoke(&call,0));
    uint64_t elapsed=cycles()-begin; model_cycles+=elapsed;
    iree_hal_buffer_view_t *output=NULL;
    CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call,&output));
    CHECK(iree_hal_device_transfer_d2h(device,iree_hal_buffer_view_buffer(output),0,&state,sizeof(state),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,iree_infinite_timeout()));
    iree_hal_buffer_view_release(output); iree_runtime_call_deinitialize(&call);
    if(state.error) { printf("FAIL operator status=%u\n",(unsigned)state.error); return 3; }
    unsigned token=0;
    for(unsigned i=0;i<512;++i) {
      float expected=expected_logits[pos*512+i],error=fabsf(state.logits[i]-expected);
      if(error>max_error) max_error=error;
      if(!isfinite(state.logits[i]) || error>0.0001f+0.00001f*fabsf(expected)) {
        printf("FAIL logits pos=%u index=%u actual_bits=%08x expected_bits=%08x\n",pos,i,bits(state.logits[i]),bits(expected)); return 4;
      }
      if(state.logits[i]>state.logits[token]) token=i;
    }
    if(token!=(unsigned)expected_tokens[pos]) { printf("FAIL token pos=%u actual=%u\n",pos,token); return 5; }
    printf("TOKEN pos=%u id=%u invoke_cycles=%lu piece=%s\n",pos,token,(unsigned long)elapsed,token_bytes+token_offsets[token]);
    for(unsigned i=0;i<512;i+=8) {
      printf("LOGITS %u %u",pos,i);
      for(unsigned j=0;j<8;++j) printf(" %08x",bits(state.logits[i+j]));
      printf("\n");
    }
    state.token=(float)token;
  }
  uint32_t z=proton_matrix_count(0x10)-zero_start,m=proton_matrix_count(0x14)-macc_start;
  if(z!=(PROTON_BACKEND?878*PROTON_STEPS:0) || m!=(PROTON_BACKEND?4072*PROTON_STEPS:0)) {
    printf("FAIL command counts zero=%u macc=%u\n",z,m); return 6;
  }
  printf("MATRIX_COUNTS zero=%u macc=%u\n",z,m);
  for(unsigned i=0;i<36;++i) {
    const proton_projection_evidence_t *e=proton_projection_evidence+i;
    printf("PROJECTION id=%u N=%u K=%u calls=%u zero=%u macc=%u\n",i,weights[i].n,weights[i].k,e->calls,e->zero,e->macc);
    unsigned ez=PROTON_BACKEND?((weights[i].n+3)/4)*PROTON_STEPS:0;
    if(e->calls!=PROTON_STEPS || e->zero!=ez || e->macc!=ez*((weights[i].k+15)/16)) return 8;
  }
  printf("MODEL steps=%u backend=%u invoke_cycles=%lu max_error_bits=%08x\n",PROTON_STEPS,PROTON_BACKEND,(unsigned long)model_cycles,bits(max_error));
#else
  CHECK(iree_runtime_call_initialize_by_name(session, iree_make_cstring_view("proton_smoke.main"), &call));
  float inputs[2][4] = {{1, -2, 3, 4}, {2, 3, -4, 0.5f}};
  iree_hal_dim_t shape[] = {4};
  for (int i = 0; i < 2; ++i) {
    iree_hal_buffer_view_t *view = NULL;
    CHECK(iree_hal_buffer_view_allocate_buffer_copy(device, iree_hal_device_allocator(device),
      1, shape, IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){.type=IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL, .usage=IREE_HAL_BUFFER_USAGE_DEFAULT},
      iree_make_const_byte_span(inputs[i], sizeof(inputs[i])), &view));
    CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, view));
    iree_hal_buffer_view_release(view);
  }
  CHECK(iree_runtime_call_invoke(&call, 0));
  iree_hal_buffer_view_t *output = NULL;
  CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &output));
  float results[4];
  CHECK(iree_hal_device_transfer_d2h(device, iree_hal_buffer_view_buffer(output), 0,
    results, sizeof(results), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  const float expected[] = {3, -8, -9, 6};
  for (int i=0; i<4; ++i) {
    printf("OUTPUT %d %d\n", i, (int)results[i]);
    if (results[i] != expected[i]) return 2;
  }
  iree_hal_buffer_view_release(output);
  iree_runtime_call_deinitialize(&call);
#endif
  iree_vm_module_release(module);
  iree_runtime_session_release(session);
  iree_hal_device_release(device);
  iree_runtime_instance_release(instance);
  if(proton_memory_end()) { printf("FAIL stack guard\n"); return 7; }
  printf("MEMORY_GUARD PASS bounded_heap exhaustion_rejected stack_canary\n");
#ifdef PROTON_MODEL
  printf("RESULT: PASS iree_model heap_peak=%lu\n", (unsigned long)proton_heap_peak);
#else
  printf("RESULT: PASS iree_smoke heap_peak=%lu\n", (unsigned long)proton_heap_peak);
#endif
  return 0;
}
