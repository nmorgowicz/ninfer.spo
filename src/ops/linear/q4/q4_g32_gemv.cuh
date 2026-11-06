#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// The q4 fp16-scale family: q4_g32_fp16 (all routes), and q4_g128_fp16 / q4_g256_fp16 (A16 routes
// only: T=1 GEMV, shared-tile GEMV, tensor-core passes, wide). A W4A8 route chosen for a coarse
// weight, or NINFER_Q4_EXACT, throws std::invalid_argument.
void q4_g32_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                     WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
