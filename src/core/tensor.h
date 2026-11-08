#pragma once

#include "core/dtype.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <source_location>

namespace ninfer {

struct Tensor {
    void* data         = nullptr;
    DType dtype        = DType::BF16;
    std::int32_t ne[4] = {1, 1, 1, 1};
    std::int64_t nb[4] = {0, 0, 0, 0};

    Tensor() noexcept = default;
    Tensor(void* data, DType dtype, std::initializer_list<std::int32_t> shape);

    std::int64_t numel() const;
    std::size_t bytes() const;
    bool is_contiguous() const;

    // A failing view, reshape, or slice names its caller and both shapes: one bad shape fails
    // the whole engine, and the message is all an operator gets back.
    Tensor view(std::initializer_list<std::int32_t> shape,
                std::source_location caller = std::source_location::current()) const;
    Tensor reshape(std::initializer_list<std::int32_t> shape,
                   std::source_location caller = std::source_location::current()) const;
    Tensor slice(int dim, std::int32_t start, std::int32_t len,
                 std::source_location caller = std::source_location::current()) const;
    Tensor permute(std::initializer_list<int> order) const;
};

} // namespace ninfer
