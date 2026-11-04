#pragma once

#include "ninfer/ops/gemma4_cache.h"
#include "ops/kernel/gemma4_fp8_kv.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Gemma-local D512 INT8-G64 layout. Code rows have 512 int8 columns, scale rows have 8 FP16
// columns (one per 64 elements), both addressed by gemma4_kv_row with identical page order.
// The numerical contract is the shared codec's: FP16-RNE(absmax/127), codes use the
// reciprocal of that represented scale, round-to-nearest, clamped to [-127, 127]. No rotation.
inline constexpr int kGemma4Int8Groups = 512 / kGemma4KvInt8Group;

// FP32 code x stored scale, rounded to BF16: the single represented-value definition shared
// by the SIMT loader and the tensor-core staging path.
__device__ __forceinline__ __nv_bfloat16 gemma4_int8_expand(std::int8_t code, __half scale) {
    return __float2bfloat16_rn(__fmul_rn(static_cast<float>(code), __half2float(scale)));
}

// One warp owns BF16 pairs lane + 32*s; slot s spans elements [64 s, 64 s + 64), which is
// exactly INT8 group s, so a 32-lane max yields that group's absmax.
// Nonfinite policy mirrors FP8: any NaN or infinity (and any group whose scale would overflow
// FP16) poisons its complete 64-element group with a NaN scale and zero codes, so the value
// reads back as NaN rather than silently becoming zero. Finite rows are the supported domain.
__device__ __forceinline__ void gemma4_int8_store(const __nv_bfloat162 (&values)[8],
                                                 std::int8_t* codes, __half* scales,
                                                 std::int64_t row, int lane) {
#pragma unroll
    for (int group = 0; group < kGemma4Int8Groups; ++group) {
        const float2 x = __bfloat1622float2(values[group]);
        const bool nonfinite = !isfinite(x.x) || !isfinite(x.y);
        float amax = fmaxf(fabsf(x.x), fabsf(x.y));
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffU, amax, offset));
        }
        const auto params = kv_cache_int8_quant_params(amax);
        const bool poison = __any_sync(0xffffffffU, nonfinite) ||
                            !isfinite(__half2float(params.scale));
        if (lane == 0) {
            scales[row * kGemma4Int8Groups + group] =
                poison ? __float2half_rn(__int_as_float(0x7fc00000)) : params.scale;
        }
        const std::int8_t c0 = poison ? std::int8_t{0}
                                      : kv_cache_int8_quant_code(x.x, params.inverse_scale);
        const std::int8_t c1 = poison ? std::int8_t{0}
                                      : kv_cache_int8_quant_code(x.y, params.inverse_scale);
        const auto packed = static_cast<std::uint16_t>(
            static_cast<std::uint16_t>(static_cast<std::uint8_t>(c0)) |
            (static_cast<std::uint16_t>(static_cast<std::uint8_t>(c1)) << 8));
        reinterpret_cast<std::uint16_t*>(codes + row * 512)[lane + 32 * group] = packed;
    }
}

} // namespace ninfer::ops::detail
