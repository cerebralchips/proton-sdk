// SPDX-License-Identifier: Apache-2.0
#include "linear.h"
#include <math.h>
#include <stddef.h>
#include <string.h>

#if defined(__riscv)
static volatile uint32_t *reg(unsigned o) { return (volatile uint32_t *)(uintptr_t)(0xe0000000UL+o); }
static void fence(void) { __asm__ volatile("fence iorw, iorw" ::: "memory"); }
static uintptr_t zero(void) { uintptr_t r; __asm__ volatile(".insn r 0x2b,0,0,%0,x0,x0":"=r"(r)::"memory"); return r; }
static uintptr_t macc(void) { uintptr_t r; __asm__ volatile(".insn r 0x2b,0,1,%0,x0,x0":"=r"(r)::"memory"); return r; }
uint32_t proton_matrix_count(unsigned o) { fence(); return *reg(o); }
int proton_matrix_probe(void) { return *reg(0)==0x4d415431 && *reg(4)==0x00010410 && !*reg(8); }
#else
uint32_t proton_matrix_count(unsigned o) { (void)o; return 0; }
int proton_matrix_probe(void) { return 0; }
#endif

int proton_qmatvec(const int8_t *x,const proton_weight_t *w,int32_t *out,int matrix) {
  if (!x || !w || !out || !w->packed || !w->n || !w->k || w->k>512 || w->n>512) return -1;
  unsigned blocks=(w->k+15)/16;
#if defined(__riscv)
  if (matrix) {
    // Other rows are zero; first milestone is batch-one autoregressive decode.
    for (unsigned i=0;i<16;++i) reg(0x100)[i]=0;
    for (unsigned n=0;n<w->n;n+=4) {
      fence(); if (zero()) return -2;
      for (unsigned b=0;b<blocks;++b) {
        uint32_t words[4]={0};
        unsigned remaining=w->k-b*16; if (remaining>16) remaining=16;
        memcpy(words,x+b*16,remaining);
        for (unsigned i=0;i<4;++i) reg(0x100)[i]=words[i];
        const int8_t *tile=w->packed+((n/4)*blocks+b)*64;
        for (unsigned i=0;i<16;++i) {
          uint32_t word; memcpy(&word,tile+i*4,4); reg(0x140)[i]=word;
        }
        fence(); if (macc()) return -3;
      }
      fence();
      for (unsigned j=0;j<4 && n+j<w->n;++j) out[n+j]=(int32_t)reg(0x180)[j];
    }
    return 0;
  }
#else
  if (matrix) return -4; // Host emulation is never reported as matrix hardware.
#endif
  for (unsigned n=0;n<w->n;++n) {
    int32_t acc=0;
    for (unsigned k=0;k<w->k;++k) {
      unsigned i=((n/4)*blocks+k/16)*64+(n%4)*16+k%16;
      acc+=(int32_t)x[k]*(int32_t)w->packed[i];
    }
    out[n]=acc;
  }
  return 0;
}

int proton_linear(const float *x,const proton_weight_t *w,float *out,int matrix) {
  if (!x || !w || !out || !w->scale || !w->k || w->k>512 || !w->n || w->n>512) return -1;
  int8_t q[512]; int32_t acc[512]; float max=0;
  for (unsigned i=0;i<w->k;++i) { if (!isfinite(x[i])) return -5; float a=fabsf(x[i]); if(a>max) max=a; }
  float scale=max>0?max/127.0f:1.0f;
  for (unsigned i=0;i<w->k;++i) { float v=x[i]/scale; q[i]=(int8_t)(int)(v+(v>=0?0.5f:-0.5f)); }
  int status=proton_qmatvec(q,w,acc,matrix); if(status) return status;
  for (unsigned i=0;i<w->n;++i) out[i]=((float)acc[i]*scale)*w->scale[i];
  return 0;
}
