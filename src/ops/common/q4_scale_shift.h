#pragma once

#include "core/weight.h"

#include <cstdint>

namespace ninfer::ops::detail {

// The q4 fp16 family keeps K/2 code bytes per row for every group size; only the scale plane
// differs: one binary16 scale per G = 32 << shift consecutive K weights. Scale index of the
// weight at column k in row r is r * (K >> (5 + shift)) + (k >> (5 + shift)).
constexpr bool is_q4_fp16_family(QType qtype) {
    return qtype == QType::Q4_G32_FP16 || qtype == QType::Q4_G128_FP16 ||
           qtype == QType::Q4_G256_FP16;
}

// 0 for g32, 2 for g128, 3 for g256; -1 for any other qtype.
constexpr int q4_scale_shift(QType qtype) {
    switch (qtype) {
    case QType::Q4_G32_FP16: return 0;
    case QType::Q4_G128_FP16: return 2;
    case QType::Q4_G256_FP16: return 3;
    default: return -1;
    }
}

} // namespace ninfer::ops::detail
