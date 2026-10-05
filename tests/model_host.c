// SPDX-License-Identifier: Apache-2.0
#include <stdio.h>
#include <math.h>
#include "model_state.h"
#include "model_data.h"
static proton_state_t state;
int main(void) {
  state.token=1;
  float maximum=0; unsigned matches=0;
  for(unsigned pos=0;pos<16;++pos) {
    state.position=(float)pos;
    proton_op((float *)&state,0,OP_EMBED,0);
    for(int l=0;l<5;++l) for(int op=OP_NORM_ATT;op<=OP_RESIDUAL_FFN;++op) proton_op((float *)&state,0,op,l);
    proton_op((float *)&state,0,OP_NORM_FINAL,0); proton_op((float *)&state,0,OP_CLASSIFIER,0);
    unsigned token=0;
    for(unsigned i=0;i<512;++i) {
      float error=fabsf(state.logits[i]-expected_logits[pos*512+i]); if(error>maximum) maximum=error;
      if(!isfinite(state.logits[i]) || error>0.0001f+0.00001f*fabsf(expected_logits[pos*512+i])) {
        printf("FAIL pos=%u logit=%u actual=%.9g expected=%.9g error=%.9g\n",pos,i,state.logits[i],expected_logits[pos*512+i],error); return 1;
      }
      if(state.logits[i]>state.logits[token]) token=i;
    }
    if(token==(unsigned)expected_tokens[pos]) ++matches;
    state.token=(float)token;
  }
  printf("HOST C quantized tokens=%u/16 max_abs_logit_error=%.9g state_bytes=%zu\n",matches,maximum,sizeof(state));
  return matches!=16 || state.error!=0;
}
