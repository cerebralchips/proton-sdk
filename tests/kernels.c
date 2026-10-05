// SPDX-License-Identifier: Apache-2.0
#include <stdio.h>
#include <math.h>
#include "linear.h"
#include "kernel_cases.h"
#include "proton/platform.h"
int main(void) {
  if(proton_memory_begin()) return 7;
  if(!proton_matrix_probe()) { printf("FAIL matrix identity\n"); return 1; }
  uint32_t z=proton_matrix_count(0x10),m=proton_matrix_count(0x14),ez=0,em=0;
  int32_t acc[512]; float y[512]; unsigned elements=0;
#ifdef PROTON_TILE_ONLY
  const unsigned count=1;
#else
  const unsigned count=36;
#endif
  for(unsigned c=0;c<count;++c) {
    if(proton_qmatvec(cases[c].q,weights+c,acc,1) || proton_linear(cases[c].x,weights+c,y,1)) return 2;
#ifdef PROTON_INJECT_FAILURE
    if(c==0) acc[0]^=1;
#endif
    for(unsigned i=0;i<weights[c].n;++i) {
      if(acc[i]!=cases[c].acc[i] || !isfinite(y[i]) || fabsf(y[i]-cases[c].y[i])>2e-5f+2e-5f*fabsf(cases[c].y[i])) {
        printf("FAIL kernel=%u element=%u acc=%d expected=%d\n",c,i,acc[i],cases[c].acc[i]); return 3;
      }
    }
    ez+=2*((weights[c].n+3)/4); em+=2*((weights[c].n+3)/4)*((weights[c].k+15)/16);
    elements+=weights[c].n;
    printf("KERNEL PASS id=%u N=%u K=%u\n",c,weights[c].n,weights[c].k);
  }
  proton_weight_t bad=weights[0]; bad.k=513;
  if(!proton_linear(cases[0].x,&bad,y,1)) return 4;
  float nonfinite[64]={NAN}; if(!proton_linear(nonfinite,weights,y,1)) return 5;
  z=proton_matrix_count(0x10)-z; m=proton_matrix_count(0x14)-m;
  if(z!=ez || m!=em) { printf("FAIL counters %u %u expected %u %u\n",z,m,ez,em); return 6; }
  printf("MATRIX_COUNTS zero=%u macc=%u\n",z,m);
  if(proton_memory_end()) return 8;
  printf("MEMORY_GUARD PASS bounded_heap exhaustion_rejected stack_canary\n");
#ifdef PROTON_TILE_ONLY
  printf("RESULT: PASS kernel_tile matrices=1 elements=%u\n",elements);
#else
  printf("RESULT: PASS kernels matrices=36 elements=%u\n",elements);
#endif
  return 0;
}
