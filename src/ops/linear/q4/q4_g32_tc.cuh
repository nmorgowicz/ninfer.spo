#pragma once

#include "ops/linear/q4/q4_operands.h"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

// Tensor-core A16 GEMV for 1..8 tokens. Weights are read once per row. `scale_shift` is
// log2(G / 32) of the weight's scale group: 0 for q4_g32, 2 for q4_g128, 3 for q4_g256.
void q4_g32_tc8(const Q4LinearOperands& operands, __nv_bfloat16* out, cudaStream_t stream,
                int scale_shift);

} // namespace ninfer::ops::detail
