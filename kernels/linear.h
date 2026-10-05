// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
#include "model_data.h"
// ABI v1: row vector times packed signed INT8 weight rows, FP32 row scales.
// Packed weights: [ceil(N/4), ceil(K/16), 4, 16]. K tails must be zero.
// No bias; dynamic activation quantization; single-hart blocking ownership.
int proton_qmatvec(const int8_t *x, const proton_weight_t *w, int32_t *out, int matrix);
int proton_linear(const float *x, const proton_weight_t *w, float *out, int matrix);
uint32_t proton_matrix_count(unsigned offset);
int proton_matrix_probe(void);
