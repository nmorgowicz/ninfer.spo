#pragma once

#include "core/weight.h"

#include <cuda_fp16.h>

#if defined(__CUDACC__)
#    define NINFER_Q4_S6_HD __host__ __device__
#else
#    define NINFER_Q4_S6_HD
#endif

#include <cstdint>

namespace ninfer::ops::detail {

// The q4 fp16 family keeps K/2 code bytes per row for every format, in the same order: 16 bytes
// per 32-weight group. Only the scale plane differs. A kernel names the plane by its scale mode:
//
//   0, 2, 3    one binary16 scale per 32 << mode weights (q4_g32, q4_g128, q4_g256). The scale
//              index of the weight at column k of row r is r * (K >> (5 + mode)) + (k >> (5 + mode)).
//   kQ4ScaleS6 two levels (q4_g32_s6_fp16): per row and per 256 columns one 8-byte little-endian
//              block, bytes 0-1 a binary16 super scale S, bytes 2-7 a 48-bit field of eight 6-bit
//              multipliers m_j, j = 0..7 the eight 32-weight groups, m_j at bits 6j..6j+5. The scale
//              of group j is the binary16 round-to-nearest-even of float(S) * float(m_j), and from
//              there on an ordinary g32 scale. A row holds K / 256 blocks, K / 32 bytes.
//
// Kernels take the mode as a template argument; the single-level instantiations keep their
// expressions, kQ4ScaleS6 adds the decode below. Scale group k of row r in mode kQ4ScaleS6 is the
// field (k & 7) of the block at byte (r * (K / 256) + (k >> 3)) * 8 of the plane.
inline constexpr int kQ4ScaleS6              = 6;
inline constexpr int kQ4S6BlockBytes         = 8;
inline constexpr int kQ4S6GroupsPerBlock     = 8;
inline constexpr int kQ4S6BlockColumns       = 256;

NINFER_Q4_S6_HD constexpr bool is_q4_fp16_family(QType qtype) {
    return qtype == QType::Q4_G32_FP16 || qtype == QType::Q4_G128_FP16 ||
           qtype == QType::Q4_G256_FP16 || qtype == QType::Q4_G32_S6_FP16;
}

// 0 for g32, 2 for g128, 3 for g256, kQ4ScaleS6 for q4_g32_s6_fp16; -1 for any other qtype.
NINFER_Q4_S6_HD constexpr int q4_scale_mode(QType qtype) {
    switch (qtype) {
    case QType::Q4_G32_FP16: return 0;
    case QType::Q4_G128_FP16: return 2;
    case QType::Q4_G256_FP16: return 3;
    case QType::Q4_G32_S6_FP16: return kQ4ScaleS6;
    default: return -1;
    }
}

NINFER_Q4_S6_HD constexpr bool q4_scale_mode_valid(int mode) {
    return mode == 0 || mode == 2 || mode == 3 || mode == kQ4ScaleS6;
}

// K columns one scale unit covers: a binary16 scale (32 << mode) or an S6 block (256). K of a
// weight in this mode is a multiple of it, except mode 0 (K is a multiple of 128 by the layout).
NINFER_Q4_S6_HD constexpr int q4_scale_span(int mode) {
    return mode == kQ4ScaleS6 ? kQ4S6BlockColumns : 32 << mode;
}

// Bytes of one row's scale plane for K columns.
NINFER_Q4_S6_HD constexpr long long q4_scale_row_bytes(int mode, long long k) {
    return mode == kQ4ScaleS6 ? k / 32 : (k >> (5 + mode)) * 2;
}

// Binary16 bits of the scale of 32-group j (0..7) of an S6 block (the 8 plane bytes as one
// little-endian word). The fp32 product is exact (11-bit significand x 6-bit multiplier); the one
// rounding is the conversion to binary16.
NINFER_Q4_S6_HD inline std::uint16_t q4_s6_scale_bits(std::uint64_t block, int j) {
    const float super      = __half2float(__ushort_as_half(static_cast<std::uint16_t>(block & 0xffffu)));
    const float multiplier = static_cast<float>(static_cast<unsigned>((block >> (16 + 6 * j)) & 63u));
    return __half_as_ushort(__float2half_rn(super * multiplier));
}

// The same scale as fp32: what an ordinary g32 kernel multiplies a code by.
NINFER_Q4_S6_HD inline float q4_s6_scale(std::uint64_t block, int j) {
    return __half2float(__ushort_as_half(q4_s6_scale_bits(block, j)));
}

// Little-endian word of the 8 bytes at p (8-byte aligned or not).
NINFER_Q4_S6_HD inline std::uint64_t q4_s6_block_word(const std::uint8_t* p) {
    std::uint64_t word = 0;
    for (int i = 0; i < kQ4S6BlockBytes; ++i) { word |= static_cast<std::uint64_t>(p[i]) << (8 * i); }
    return word;
}

} // namespace ninfer::ops::detail
