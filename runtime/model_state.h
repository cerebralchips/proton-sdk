// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stddef.h>
#define MODEL_CONTEXT 64
typedef struct {
  float token, position, backend, error;
  float x[64], xb[64], tmp[64], q[64], k[32], v[32], up[172], gate[172], logits[512];
  float keys[5][MODEL_CONTEXT][32], values[5][MODEL_CONTEXT][32];
} proton_state_t;
enum { OP_EMBED, OP_NORM_ATT, OP_Q, OP_K, OP_V, OP_ATTENTION, OP_O,
       OP_RESIDUAL_ATT, OP_NORM_FFN, OP_UP, OP_GATE, OP_SILU, OP_DOWN,
       OP_RESIDUAL_FFN, OP_NORM_FINAL, OP_CLASSIFIER };
void proton_op(float *base, size_t offset, int op, int layer);
typedef struct { unsigned calls, zero, macc; } proton_projection_evidence_t;
extern proton_projection_evidence_t proton_projection_evidence[36];
