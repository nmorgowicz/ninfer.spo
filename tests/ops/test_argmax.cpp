#include "ninfer/ops/argmax.h"
#include "core/device.h"
#include <algorithm>
#include "ops/op_tester.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

std::vector<std::int32_t> argmax_oracle(const std::vector<std::uint16_t>& logits,
                                        std::int32_t physical_rows, std::int32_t tokens,
                                        std::int32_t valid_rows) {
    std::vector<std::int32_t> expected(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t base = static_cast<std::size_t>(token) * physical_rows;
        std::int32_t best      = 0;
        float best_value       = bf16_to_f32(logits[base]);
        for (std::int32_t row = 1; row < valid_rows; ++row) {
            const float value = bf16_to_f32(logits[base + row]);
            if (value > best_value) {
                best       = row;
                best_value = value;
            }
        }
        expected[static_cast<std::size_t>(token)] = best;
    }
    return expected;
}

std::vector<std::uint16_t> make_logits(std::int32_t physical_rows, std::int32_t tokens,
                                       std::int32_t valid_rows) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t base = static_cast<std::size_t>(token) * physical_rows;
        for (std::int32_t row = 0; row < physical_rows; ++row) {
            const std::uint32_t mixed = static_cast<std::uint32_t>(row) * 1664525u +
                                        static_cast<std::uint32_t>(token + 1) * 1013904223u;
            const float value  = -24.0f + static_cast<float>(mixed % 3072u) * (1.0f / 256.0f);
            logits[base + row] = f32_to_bf16(value);
        }

        std::int32_t first  = 17 + token * 7919;
        std::int32_t second = valid_rows - 1 - token * 65537;
        first %= valid_rows;
        second %= valid_rows;
        if (second < 0) { second += valid_rows; }
        if (first == second) { second = (second + 1) % valid_rows; }
        if (second < first) {
            const std::int32_t temporary = first;
            first                        = second;
            second                       = temporary;
        }
        logits[base + first]  = f32_to_bf16(token % 3 == 0 ? -0.5f : 32.0f + static_cast<float>(token));
        logits[base + second] = logits[base + first];

        if (token == 0) {
            std::fill(logits.begin() + base, logits.begin() + base + valid_rows, f32_to_bf16(-4.0f));
        }
        if (valid_rows < physical_rows) {
            logits[base + valid_rows]        = f32_to_bf16(32768.0f);
            logits[base + physical_rows - 1] = f32_to_bf16(65536.0f);
        }
    }
    return logits;
}

int run_case(std::int32_t physical_rows, std::int32_t valid_rows, std::int32_t tokens) {
    auto logits   = make_logits(physical_rows, tokens, valid_rows);
    auto expected = argmax_oracle(logits, physical_rows, tokens, valid_rows);

    GuardedDeviceBuffer device_logits(logits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_output(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
    device_logits.copy_from_host(logits.data(), logits.size() * sizeof(std::uint16_t));
    device_output.fill(0xcd);

    Tensor logits_tensor(device_logits.data(), DType::BF16, {physical_rows, tokens});
    Tensor output_tensor(device_output.data(), DType::I32, {tokens});
    ops::argmax(logits_tensor, output_tensor, valid_rows, nullptr);
    cuda_synchronize();

    if (physical_rows == 248320 && tokens == 128) {
        cudaStream_t stream;
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        ops::argmax(logits_tensor, output_tensor, valid_rows, stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int t = 0; t < tokens; ++t)
            logits[static_cast<std::size_t>(t) * physical_rows + valid_rows - 1] = f32_to_bf16(8192.0f);
        CUDA_CHECK(cudaMemcpyAsync(device_logits.data(), logits.data(), device_logits.bytes(),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaStreamDestroy(stream));
        expected = argmax_oracle(logits, physical_rows, tokens, valid_rows);
    }

    const auto actual =
        from_device<std::int32_t>(device_output.data(), static_cast<std::size_t>(tokens));
    const auto logits_after = from_device<std::uint16_t>(device_logits.data(), logits.size());
    const std::string label = "argmax rows=" + std::to_string(physical_rows) +
                              " valid=" + std::to_string(valid_rows) +
                              " T=" + std::to_string(tokens);

    int failures = 0;
    failures += verify_exact(label.c_str(), actual, expected);
    failures += verify_exact((label + " preserves logits").c_str(), logits_after, logits);
    failures += device_logits.verify_guards((label + " logits").c_str());
    failures += device_output.verify_guards((label + " output").c_str());
    return failures;
}

// A draft column of NaN logits must still yield an id from the shortlist, never an index past it.
int run_shortlist_nan_case() {
    constexpr std::int32_t kRows   = 1024;
    constexpr std::int32_t kHidden = 64;
    constexpr std::int32_t kTokens = 3;
    constexpr std::int32_t kCandidates = (kRows / 512) * 2;

    std::vector<std::uint16_t> hidden(static_cast<std::size_t>(kHidden) * kTokens);
    for (std::size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] = f32_to_bf16(static_cast<float>(i % 7) * 0.125f - 0.25f);
    }
    std::vector<std::uint16_t> head(static_cast<std::size_t>(kRows) * kHidden);
    for (std::size_t i = 0; i < head.size(); ++i) {
        head[i] = f32_to_bf16(static_cast<float>((i * 2654435761u) % 97u) / 97.0f - 0.5f);
    }
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(kRows) * kTokens);
    for (std::int32_t token = 0; token < kTokens; ++token) {
        for (std::int32_t row = 0; row < kRows; ++row) {
            logits[static_cast<std::size_t>(token) * kRows + row] =
                token == 1 ? std::uint16_t{0x7fc0} : f32_to_bf16(static_cast<float>(row % 31));
        }
    }
    std::vector<std::int32_t> ids(kRows);
    for (std::int32_t row = 0; row < kRows; ++row) { ids[static_cast<std::size_t>(row)] = row; }

    GuardedDeviceBuffer device_hidden(hidden.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_head(head.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_logits(logits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_ids(ids.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_candidate_ids(kCandidates * kTokens * sizeof(std::int32_t));
    GuardedDeviceBuffer device_candidate_scores(kCandidates * kTokens * sizeof(float));
    GuardedDeviceBuffer device_output(kTokens * sizeof(std::int32_t));
    device_hidden.copy_from_host(hidden.data(), hidden.size() * sizeof(std::uint16_t));
    device_head.copy_from_host(head.data(), head.size() * sizeof(std::uint16_t));
    device_logits.copy_from_host(logits.data(), logits.size() * sizeof(std::uint16_t));
    device_ids.copy_from_host(ids.data(), ids.size() * sizeof(std::int32_t));
    device_output.fill(0xcd);

    Weight exact_head;
    exact_head.qtype  = QType::BF16;
    exact_head.layout = QuantLayout::Contiguous;
    exact_head.qdata  = device_head.data();
    exact_head.n      = kRows;
    exact_head.k      = kHidden;
    Tensor hidden_tensor(device_hidden.data(), DType::BF16, {kHidden, kTokens});
    Tensor logits_tensor(device_logits.data(), DType::BF16, {kRows, kTokens});
    Tensor candidate_ids(device_candidate_ids.data(), DType::I32, {kCandidates, kTokens});
    Tensor candidate_scores(device_candidate_scores.data(), DType::FP32, {kCandidates, kTokens});
    Tensor output_tensor(device_output.data(), DType::I32, {kTokens});
    ops::shortlist_exact_argmax(hidden_tensor, logits_tensor, kRows, exact_head,
                                static_cast<const std::int32_t*>(device_ids.data()), candidate_ids,
                                candidate_scores, output_tensor, nullptr);
    cuda_synchronize();

    int failures       = 0;
    const auto actual  = from_device<std::int32_t>(device_output.data(), kTokens);
    for (std::int32_t token = 0; token < kTokens; ++token) {
        const std::int32_t id = actual[static_cast<std::size_t>(token)];
        if (id < 0 || id >= kRows) {
            std::cerr << "shortlist NaN column: token " << token << " got id " << id << '\n';
            ++failures;
        }
    }
    failures += device_ids.verify_guards("shortlist NaN id map");
    failures += device_output.verify_guards("shortlist NaN output");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_case(248320, 248077, 1);
    failures += run_case(248320, 248077, 2);
    failures += run_case(248320, 248077, 8);
    failures += run_case(248320, 248077, 9);
    failures += run_case(248320, 248077, 64);
    failures += run_case(248320, 248077, 15);
    failures += run_case(248320, 248077, 128);
    failures += run_case(131072, 131072, 1);
    failures += run_case(131072, 131072, 15);
    failures += run_case(131072, 131072, 120);
    failures += run_shortlist_nan_case();
    std::cout << (failures ? "FAIL" : "OK") << " argmax\n";
    return failures ? 1 : 0;
}
