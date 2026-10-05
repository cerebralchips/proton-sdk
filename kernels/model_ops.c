// SPDX-License-Identifier: Apache-2.0
#include "model_state.h"
#include "linear.h"
#include <math.h>
#include <string.h>
proton_projection_evidence_t proton_projection_evidence[36];

static void norm(float *out,const float *x,const float *w) {
  float sum=0; for(unsigned i=0;i<64;++i) sum+=x[i]*x[i];
  float scale=1.0f/sqrtf(sum/64.0f+1e-5f);
  for(unsigned i=0;i<64;++i) out[i]=w[i]*(scale*x[i]);
}
static void attention(proton_state_t *s,unsigned l,unsigned pos) {
  static const float freq[4]={1.0f,.1f,.01f,.001f};
  for(unsigned i=0;i<64;i+=2) {
    float c=cosf((float)pos*freq[(i%8)/2]), v=sinf((float)pos*freq[(i%8)/2]);
    float a=s->q[i],b=s->q[i+1]; s->q[i]=a*c-b*v; s->q[i+1]=a*v+b*c;
    if(i<32) { a=s->k[i]; b=s->k[i+1]; s->k[i]=a*c-b*v; s->k[i+1]=a*v+b*c; }
  }
  memcpy(s->keys[l][pos],s->k,sizeof(s->k)); memcpy(s->values[l][pos],s->v,sizeof(s->v));
  for(unsigned h=0;h<8;++h) {
    float scores[MODEL_CONTEXT],max=-INFINITY,sum=0;
    for(unsigned t=0;t<=pos;++t) {
      float a=0; for(unsigned i=0;i<8;++i) a+=s->q[h*8+i]*s->keys[l][t][(h/2)*8+i];
      scores[t]=a/sqrtf(8.0f); if(scores[t]>max) max=scores[t];
    }
    for(unsigned t=0;t<=pos;++t) { scores[t]=expf(scores[t]-max); sum+=scores[t]; }
    for(unsigned i=0;i<8;++i) s->xb[h*8+i]=0;
    for(unsigned t=0;t<=pos;++t) {
      float a=scores[t]/sum;
      for(unsigned i=0;i<8;++i) s->xb[h*8+i]+=a*s->values[l][t][(h/2)*8+i];
    }
  }
}
void proton_op(float *base,size_t offset,int op,int layer) {
  proton_state_t *s=(proton_state_t *)(base+offset);
  if(s->error) return;
  if(layer<0 || layer>=5 || s->position<0 || s->position>=MODEL_CONTEXT ||
     s->token<0 || s->token>=512 || (s->backend!=0 && s->backend!=1)) { s->error=1; return; }
  int matrix=(int)s->backend,status=0;
  const proton_weight_t *w=weights+layer*7;
  int projection=-1;
  switch(op) {
  case OP_Q: projection=layer*7; break;
  case OP_K: projection=layer*7+1; break;
  case OP_V: projection=layer*7+2; break;
  case OP_O: projection=layer*7+3; break;
  case OP_UP: projection=layer*7+4; break;
  case OP_GATE: projection=layer*7+5; break;
  case OP_DOWN: projection=layer*7+6; break;
  case OP_CLASSIFIER: projection=35; break;
  }
  uint32_t z=0,m=0;
  if(projection>=0) { z=proton_matrix_count(0x10); m=proton_matrix_count(0x14); }
  switch(op) {
  case OP_EMBED: memcpy(s->x,embedding+(unsigned)s->token*64,sizeof(s->x)); break;
  case OP_NORM_ATT: norm(s->xb,s->x,norm_att+layer*64); break;
  case OP_Q: status=proton_linear(s->xb,w+0,s->q,matrix); break;
  case OP_K: status=proton_linear(s->xb,w+1,s->k,matrix); break;
  case OP_V: status=proton_linear(s->xb,w+2,s->v,matrix); break;
  case OP_ATTENTION: attention(s,layer,(unsigned)s->position); break;
  case OP_O: status=proton_linear(s->xb,w+3,s->tmp,matrix); break;
  case OP_RESIDUAL_ATT: for(unsigned i=0;i<64;++i) s->x[i]+=s->tmp[i]; break;
  case OP_NORM_FFN: norm(s->xb,s->x,norm_ffn+layer*64); break;
  case OP_UP: status=proton_linear(s->xb,w+4,s->up,matrix); break;
  case OP_GATE: status=proton_linear(s->xb,w+5,s->gate,matrix); break;
  case OP_SILU: for(unsigned i=0;i<172;++i) s->up[i]=(s->up[i]*(1.0f/(1.0f+expf(-s->up[i]))))*s->gate[i]; break;
  case OP_DOWN: status=proton_linear(s->up,w+6,s->xb,matrix); break;
  case OP_RESIDUAL_FFN: for(unsigned i=0;i<64;++i) s->x[i]+=s->xb[i]; break;
  case OP_NORM_FINAL: norm(s->x,s->x,norm_final); break;
  case OP_CLASSIFIER: status=proton_linear(s->x,weights+35,s->logits,matrix); break;
  default: status=-6;
  }
  if(status) s->error=(float)-status;
  if(projection>=0) {
    proton_projection_evidence_t *e=proton_projection_evidence+projection;
    ++e->calls; e->zero+=proton_matrix_count(0x10)-z; e->macc+=proton_matrix_count(0x14)-m;
  }
}
