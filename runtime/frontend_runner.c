// SPDX-License-Identifier: Apache-2.0
// One invocation only: token=BOS, position=0, explicit zero KV inputs.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "iree/runtime/api.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/hal/local/loaders/static_library_loader.h"
#include "iree/vm/bytecode/module.h"
#include "proton/platform.h"
#include "libraries.h"
#include "reference.h"
extern const unsigned char module_data[];
extern const size_t module_size;
extern uintptr_t proton_heap_peak;
static float cache[5*64*32];
static float actual[5*64*32];
static uint64_t cycles(void) { uint64_t x; __asm__ volatile("rdcycle %0":"=r"(x)); return x; }
static unsigned bits(float x) { unsigned b; memcpy(&b,&x,4); return b; }
#define CHECK(expr) do { iree_status_t s = (expr); if (!iree_status_is_ok(s)) { \
  printf("FAIL IREE line=%d status=%d\n",__LINE__,(int)iree_status_code(s)); \
  iree_status_ignore(s); return 1; } } while(0)

// Lossless FP32 bit output except signed zero, encoded in explicit zero runs.
// Every index is covered, allowing independent full-tensor host comparison.
static void dump(const char *name, const float *data, unsigned count) {
  for(unsigned i=0;i<count;) {
    unsigned n=0;
    if(data[i]==0.0f) {
      while(i+n<count && data[i+n]==0.0f) ++n;
      printf("ZERO %s %u %u\n",name,i,n);
    } else {
      while(i+n<count && data[i+n]!=0.0f && n<8) ++n;
      printf("DATA %s %u %u",name,i,n);
      for(unsigned j=0;j<n;++j) printf(" %08x",bits(data[i+j]));
      printf("\n");
    }
    i+=n;
  }
}

int main(void) {
  printf("FRONTEND START precision=%s steps=1\n",FRONTEND_PRECISION);
  if(proton_memory_begin()) { printf("FAIL memory initialization\n"); return 1; }
  iree_allocator_t alloc=iree_allocator_system();
  iree_runtime_instance_options_t io;
  iree_runtime_instance_options_initialize(&io);
  iree_runtime_instance_t *instance=NULL;
  CHECK(iree_runtime_instance_create(&io,alloc,&instance));
  iree_hal_executable_loader_t *loader=NULL;
  CHECK(iree_hal_static_library_loader_create(IREE_ARRAYSIZE(proton_libraries),proton_libraries,
    iree_hal_executable_import_provider_null(),alloc,&loader));
  iree_hal_allocator_t *device_alloc=NULL;
  CHECK(iree_hal_allocator_create_heap(iree_make_cstring_view("proton"),alloc,alloc,&device_alloc));
  iree_hal_sync_device_params_t params;
  iree_hal_sync_device_params_initialize(&params);
  iree_hal_device_t *device=NULL;
  CHECK(iree_hal_sync_device_create(iree_make_cstring_view("local-sync"),&params,1,&loader,device_alloc,alloc,&device));
  iree_hal_executable_loader_release(loader);
  iree_hal_allocator_release(device_alloc);
  iree_runtime_session_options_t so;
  iree_runtime_session_options_initialize(&so);
  iree_runtime_session_t *session=NULL;
  CHECK(iree_runtime_session_create_with_device(instance,&so,device,alloc,&session));
  iree_vm_module_t *module=NULL;
  CHECK(iree_vm_bytecode_module_create(iree_runtime_instance_vm_instance(instance),0,
    iree_make_const_byte_span(module_data,module_size),iree_allocator_null(),alloc,&module));
  CHECK(iree_runtime_session_append_module(session,module));
  iree_runtime_call_t call;
  CHECK(iree_runtime_call_initialize_by_name(session,iree_make_cstring_view(FRONTEND_FUNCTION),&call));
  int64_t token=1,position=0;
  iree_hal_dim_t scalar_shape[]={1}, cache_shape[]={5,64,32};
  const void *inputs[]={&token,&position,cache,cache};
  for(unsigned i=0;i<4;++i) {
    iree_hal_buffer_view_t *view=NULL;
    CHECK(iree_hal_buffer_view_allocate_buffer_copy(device,iree_hal_device_allocator(device),
      i<2?1:3,i<2?scalar_shape:cache_shape,
      i<2?IREE_HAL_ELEMENT_TYPE_SINT_64:IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){.type=IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,.usage=IREE_HAL_BUFFER_USAGE_DEFAULT},
      iree_make_const_byte_span(inputs[i],i<2?sizeof(int64_t):sizeof(cache)),&view));
    CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call,view));
    iree_hal_buffer_view_release(view);
  }
  printf("FRONTEND INVOKE steps=1\n");
  uint64_t begin=cycles();
  CHECK(iree_runtime_call_invoke(&call,0)); // Exactly one decoder invocation.
  uint64_t elapsed=cycles()-begin;
  const char *names[]={"logits","keys","values"};
  const float *expected[]={reference_logits,reference_keys,reference_values};
  unsigned next_token=0;
  for(unsigned t=0;t<3;++t) {
    unsigned count=t?10240:512;
    iree_hal_buffer_view_t *view=NULL;
    CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call,&view));
    if(iree_hal_buffer_view_element_type(view)!=IREE_HAL_ELEMENT_TYPE_FLOAT_32 ||
       iree_hal_buffer_view_byte_length(view)!=count*sizeof(float)) {
      printf("FAIL output type or length tensor=%s\n",names[t]); return 2;
    }
    const iree_hal_dim_t *dims=iree_hal_buffer_view_shape_dims(view);
    if(iree_hal_buffer_view_shape_rank(view)!=(t?3:1) ||
       (t?(dims[0]!=5 || dims[1]!=64 || dims[2]!=32):(dims[0]!=512))) {
      printf("FAIL output shape tensor=%s\n",names[t]); return 2;
    }
    CHECK(iree_hal_device_transfer_d2h(device,iree_hal_buffer_view_buffer(view),0,actual,count*sizeof(float),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,iree_infinite_timeout()));
    iree_hal_buffer_view_release(view);
    float max_error=0;
    for(unsigned i=0;i<count;++i) {
      float error=fabsf(actual[i]-expected[t][i]);
      if(!isfinite(actual[i]) || error>0.0001f+0.00001f*fabsf(expected[t][i])) {
        printf("FAIL tensor=%s index=%u actual=%08x expected=%08x\n",names[t],i,bits(actual[i]),bits(expected[t][i])); return 3;
      }
      if(error>max_error) max_error=error;
      if(t==0 && actual[i]>actual[next_token]) next_token=i;
    }
    dump(names[t],actual,count);
    printf("TENSOR PASS name=%s elements=%u max_error_bits=%08x\n",names[t],count,bits(max_error));
  }
  if(next_token!=REFERENCE_TOKEN) { printf("FAIL token=%u\n",next_token); return 4; }
  iree_runtime_call_deinitialize(&call);
  iree_vm_module_release(module);
  iree_runtime_session_release(session);
  iree_hal_device_release(device);
  iree_runtime_instance_release(instance);
  if(proton_memory_end()) { printf("FAIL stack guard\n"); return 5; }
  printf("TOKEN pos=0 id=%u\n",next_token);
  printf("FRONTEND steps=1 precision=%s invoke_cycles=%lu\n",FRONTEND_PRECISION,(unsigned long)elapsed);
  printf("MEMORY_GUARD PASS bounded_heap exhaustion_rejected stack_canary\n");
  printf("RESULT: PASS frontend_%s heap_peak=%lu\n",FRONTEND_PRECISION,(unsigned long)proton_heap_peak);
  return 0;
}
