#pragma once

#include "ninfer/ops/gemma4_cache.h"
#include "ninfer/types/engine.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::models::gemma4 {

[[nodiscard]] inline ops::Gemma4KvFormat gemma4_kv_format(KvCacheStorage requested) {
    switch (requested) {
    case KvCacheStorage::BFloat16: return ops::Gemma4KvFormat::BFloat16;
    case KvCacheStorage::Fp8E4M3Row256: return ops::Gemma4KvFormat::Fp8E4m3Row256;
    default: throw std::invalid_argument("gemma4 KV cache supports only bf16 and fp8");
    }
}

struct KvLayerLayout {
    ops::Gemma4KvFormat format = ops::Gemma4KvFormat::BFloat16;
    DType code_dtype = DType::BF16;
    int scale_groups = 0;
    std::size_t code_page_bytes = 0;  // One K or V plane.
    std::size_t scale_page_bytes = 0;

    [[nodiscard]] std::size_t page_bytes() const noexcept {
        return 2U * (code_page_bytes + scale_page_bytes);
    }
};

// Sliding layers always retain BF16, independently of the requested global format.
[[nodiscard]] inline KvLayerLayout kv_layer_layout(KvCacheStorage requested, bool global,
                                                    int head_dim, int kv_heads) {
    const auto requested_format = gemma4_kv_format(requested);
    if (head_dim <= 0 || kv_heads <= 0) {
        throw std::invalid_argument("gemma4 KV cache geometry is invalid");
    }
    KvLayerLayout layout;
    layout.format = global ? requested_format : ops::Gemma4KvFormat::BFloat16;
    const auto rows = static_cast<std::size_t>(ops::kGemma4KvPageTokens) * kv_heads;
    if (layout.format == ops::Gemma4KvFormat::Fp8E4m3Row256) {
        if (head_dim != 512) {
            throw std::invalid_argument("gemma4 fp8 global KV requires head dimension 512");
        }
        layout.code_dtype = DType::U8;
        layout.scale_groups = head_dim / ops::kGemma4KvFp8Group;
        layout.code_page_bytes = rows * head_dim;
        layout.scale_page_bytes = rows * layout.scale_groups * sizeof(std::uint16_t);
    } else {
        layout.code_page_bytes = rows * head_dim * sizeof(std::uint16_t);
    }
    return layout;
}

inline void validate_native_context(std::uint64_t requested, std::uint64_t native) {
    if (native == 0 || requested > native) {
        throw std::invalid_argument("gemma4 max_context exceeds artifact max_position_embeddings");
    }
}

struct KvPageCapacity {
    std::uint32_t pages = 0;
    std::uint32_t tokens = 0;
    std::uint32_t extra_tokens = 0;
};

// Fit is already net of BF16 rings, snapshots, workspace and resolved headroom. Explicit
// requests round UP before testing fit; spare-pool accounting starts at these actual pages.
[[nodiscard]] inline KvPageCapacity kv_page_capacity(const KvCapacityPolicy& policy,
                                                     std::uint32_t max_context,
                                                     std::uint64_t native_context,
                                                     std::uint64_t fit_tokens,
                                                     std::uint32_t slots) {
    validate_native_context(max_context, native_context);
    constexpr std::uint64_t page = ops::kGemma4KvPageTokens;
    std::uint64_t pages = 0;
    const std::uint64_t fit_pages = fit_tokens / page;
    if (policy.mode == KvCapacityMode::Explicit) {
        if (policy.explicit_tokens == 0) {
            throw std::invalid_argument("gemma4 explicit KV capacity must be nonzero");
        }
        pages = (static_cast<std::uint64_t>(policy.explicit_tokens) + page - 1U) / page;
        if (policy.explicit_tokens > max_context || pages > fit_pages) {
            throw std::invalid_argument("gemma4 explicit KV capacity does not fit context or device memory");
        }
    } else if (policy.mode == KvCapacityMode::Automatic) {
        pages = std::min<std::uint64_t>(max_context / page, fit_pages);
    } else {
        throw std::invalid_argument("gemma4 KV capacity mode is invalid");
    }
    if (pages == 0 || pages * page > native_context ||
        pages * page > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("gemma4 page-aligned KV capacity does not fit native context or device memory");
    }
    const auto tokens = pages * page;
    const auto spare_pages = std::min<std::uint64_t>({pages * slots, 65536U / page,
                                                     fit_pages - pages});
    return {static_cast<std::uint32_t>(pages), static_cast<std::uint32_t>(tokens),
            static_cast<std::uint32_t>(spare_pages * page)};
}

} // namespace ninfer::models::gemma4
