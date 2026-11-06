#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// The q4 fp16-scale family: q4_g32_fp16, q4_g128_fp16, q4_g256_fp16 and q4_g32_s6_fp16, on every
// route (A16 GEMV, tensor-core and wide kernels, and the Fused / Baked / Exact W4A8 routes of q4_g32_route.h).
void q4_g32_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                     WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
