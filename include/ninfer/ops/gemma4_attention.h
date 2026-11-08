#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/gemma4_cache.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Workspace gemma4_attention takes from the arena for any call with at most T query tokens.
// Nonzero only when the grid is too small to fill the device and keys are split across blocks.
// D512 tensor-core splits reserve the context-independent bound before graph capture.
// The active device and runtime switches must be selected before sizing/capture.
[[nodiscard]] std::size_t gemma4_attention_workspace_capacity_bytes(std::int32_t head_dim,
                                                                    std::int32_t q_heads,
                                                                    std::int32_t kv_heads,
                                                                    std::int32_t tokens);

/**
 * Causal grouped-query attention over a Gemma paged KV cache.
 *
 *   score[j] = scale * dot(q[:,h,i], k[:,kv,j])
 *   out[:,h,i] = softmax(score) * v[:,kv,j]
 *
 * Key j is visible to query position p when j <= p and (window == 0 or p - j < window).
 * Query head h reads KV head floor(h / (Hq/Hkv)). Admitted profiles are D=256 with any
 * positive Hq divisible by Hkv, and D=512 with the same rule. Gemma 4 uses Hq=32, Hkv=16
 * (sliding, window 1024) and Hq=32, Hkv=4 (global, window 0), both with scale 1.
 *
 * q and out are contiguous BF16 [D,Hq,T]. positions is contiguous I32 [T].
 * cache follows Gemma4CacheView's plane layout. FP8 is accepted only for D512/global
 * (window zero); each code is expanded with an FP32 code-times-stored-FP16-scale
 * product rounded to BF16, identically for SIMT and tensor-core staging. INT8-G64 is also
 * accepted only for D512/global: the same expansion applies with one FP16 scale per 64
 * elements ([8,64,Hkv,pages] scale planes); FP8 QK never combines with INT8.
 * block_table maps absolute position p to physical page block_table[p/64], slot p%64.
 * The Op writes all of out. Small SIMT grids and underfilled D512 tensor-core grids
 * split visible keys; FP32 partial outputs and maxima/sums merge in a fixed order.
 * Splitting changes FP32 reduction order, not QK precision, and is not bit-exact to
 * the unsplit walk. FP8 QK still requires its explicit cache descriptor selection.
 */
void gemma4_attention(const Tensor& q, const Gemma4CacheView& cache,
                      const Tensor& positions, int window, float scale,
                      Tensor& out, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
