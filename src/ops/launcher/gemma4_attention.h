#pragma once

#include "core/tensor.h"
#include "ninfer/ops/gemma4_cache.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// From one full 64-row query tile on, prefill goes through the tensor-core kernel.
inline constexpr std::int32_t kGemma4AttentionMmaMinTokens = 64;

// Number of query rows in one D=512 tensor-core query tile.
inline constexpr std::int32_t kGemma4AttentionD512TileRows = 64;

// Number of context split-K segments the D=512 tensor-core kernel will use; 1 means the
// single-kernel path. pages <= 0 returns the context-unclamped bound used for workspace
// sizing. Honors NINFER_GEMMA4_ATTN_SPLIT_K.
int gemma4_attention_mma_d512_segments(std::int32_t tokens, std::int32_t q_heads,
                                       std::int32_t pages, std::int32_t window);

// Number of key chunks per query; 1 means no split and no workspace.
int gemma4_attention_chunks(std::int32_t tokens, std::int32_t kv_heads);

// partial is FP32 [D, parts, Hq, T] and stats FP32 [2, parts, Hq, T], unused for parts=1.
// parts is gemma4_attention_chunks for SIMT or gemma4_attention_mma_d512_segments for D512 MMA.
void gemma4_attention_launch(const Tensor& q, const Gemma4CacheView& cache,
                             const Tensor& positions, int window, float scale, Tensor& out,
                             float* partial, float* stats, cudaStream_t stream);

// Tensor-core path for wide prefill chunks (D=256 or 512). The D=512 kernel may split the
// context into segments and then needs the same partial/stats workspace as above.
void gemma4_attention_mma_launch(const Tensor& q, const Gemma4CacheView& cache,
                                 const Tensor& positions, int window, float scale, Tensor& out,
                                 float* partial, float* stats, cudaStream_t stream);

// The D=512 kernel behind gemma4_attention_mma_launch.
void gemma4_attention_mma_d512_launch(const Tensor& q, const Gemma4CacheView& cache,
                                      const Tensor& positions, int window, float scale,
                                      Tensor& out, float* partial, float* partial_stats,
                                      cudaStream_t stream);

} // namespace ninfer::ops::detail
