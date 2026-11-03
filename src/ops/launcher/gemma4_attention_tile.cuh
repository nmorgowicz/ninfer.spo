#pragma once

#include "ops/common/memory.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kernel/gemma4_fp8_kv.cuh"
#include "ops/softmax_attention/common/causal_tile_io.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// One [Bc, Width] tile of keys [k0, k0 + Bc) from columns [column, column + Width) of the
// [D,64,Hkv,pages] cache into swizzled shared memory. Keys past `last` are zeroed so the
// unwritten cache tail never reaches the tensor cores.
template <int D, int Width, int Bc, int Threads>
__device__ __forceinline__ void gemma4_stage_keys(__nv_bfloat16* dst, const __nv_bfloat16* cache,
                                                  int kv_heads, int kv_head, int physical, int k0,
                                                  int last, int column, int tid) {
    constexpr int VecPerRow = Width / 8;
    // Form addresses only for mapped rows, not negative physical page IDs.
#pragma unroll
    for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
        const int key    = chunk / VecPerRow;
        const int d      = (chunk % VecPerRow) * 8;
        __nv_bfloat16* p = &dst[key * Width + causal_swizzle(key, d)];
        if (physical >= 0 && k0 + key <= last) {
            const auto row = gemma4_kv_row(physical, kv_heads, kv_head,
                                           (k0 & kPagedKVPageMask) + key);
            cp_async<16, Cache::cg>(p, cache + row * D + column + d);
        } else {
            store_vec(p, make_int4(0, 0, 0, 0));
        }
    }
}

// Direct vector code loads, expanded synchronously to the same BF16 shared tile.
// Callers retain their existing block barriers before any cross-thread consumption.
template <int Width, int Bc, int Threads>
__device__ __forceinline__ void gemma4_stage_fp8_keys(
    __nv_bfloat16* dst, const std::uint8_t* codes, const __half* scales,
    int kv_heads, int kv_head, int physical, int k0, int last, int column, int tid) {
    constexpr int VecPerRow = Width / 8;
#pragma unroll
    for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
        const int key = chunk / VecPerRow;
        const int d = column + (chunk % VecPerRow) * 8;
        __nv_bfloat16* p = dst + key * Width + causal_swizzle(key, d - column);
        if (physical >= 0 && k0 + key <= last) {
            const auto row = gemma4_kv_row(physical, kv_heads, kv_head,
                                           (k0 & kPagedKVPageMask) + key);
            const uint2 packed = *reinterpret_cast<const uint2*>(codes + row * 512 + d);
            const auto* bytes = reinterpret_cast<const std::uint8_t*>(&packed);
            const __half scale = scales[row * 2 + d / kGemma4KvFp8Group];
#pragma unroll
            for (int i = 0; i < 8; ++i) { p[i] = gemma4_fp8_expand(bytes[i], scale); }
        } else {
            store_vec(p, make_int4(0, 0, 0, 0));
        }
    }
}

} // namespace ninfer::ops::detail
