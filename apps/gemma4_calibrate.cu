// Offline eager-only collection. Production serving never instantiates this sink.
#include "core/device.h"
#include "models/gemma4/load.h"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_bf16.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::models::gemma4;
using Json = nlohmann::json;
constexpr int kBlocks = 128;
constexpr int kThreads = 256;
constexpr int kSamplesPerCall = 8;
constexpr int kSampleCapacity = 1024;
constexpr int kProbeCount = 3;
// Unlike the power-of-two probes, 1.2F exercises product rounding for BF16 inputs.
__host__ __device__ constexpr float probe_divisor(int index) {
    return index == 0 ? 1.0F : index == 1 ? 256.0F : 1.2F;
}
constexpr std::array<float, kProbeCount> kProbeDivisors = {
    probe_divisor(0), probe_divisor(1), probe_divisor(2)};

// Fixed seed and integer algorithm make the offline sample reproducible.
__host__ __device__ std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct Sample {
    float values[16];
    std::uint32_t lo[kProbeCount], hi[kProbeCount];
    std::uint8_t scales[kProbeCount];
    std::uint64_t call, block;
};

__global__ void sample_inputs(const __nv_bfloat16* values, std::uint64_t blocks,
                              std::uint64_t call, Sample* samples) {
    const unsigned slot = threadIdx.x;
    const unsigned count = blocks < kSamplesPerCall ? static_cast<unsigned>(blocks) : kSamplesPerCall;
    if (slot >= count) { return; }
    const auto begin = blocks * slot / count;
    const auto end = blocks * (slot + 1) / count;
    const auto block = begin + mix(call * kSamplesPerCall + slot) % (end - begin);
    auto& sample = samples[slot];
    sample.call = call;
    sample.block = block;
    for (int i = 0; i < 16; ++i) { sample.values[i] = __bfloat162float(values[block * 16 + i]); }
    // These come from the serving codec, not a second implementation of its rounding rules.
    for (int i = 0; i < kProbeCount; ++i) {
        const auto encoded = ops::detail::quantize_nvfp4_k16(values + block * 16, probe_divisor(i));
        sample.lo[i] = encoded.codes_lo;
        sample.hi[i] = encoded.codes_hi;
        sample.scales[i] = encoded.scale;
    }
}

// Absmax is reduction-order independent. A NaN/Inf is recorded separately, never ignored by max.
__global__ void input_absmax(const __nv_bfloat16* values, std::int64_t size,
                             float* maxima, unsigned* invalid) {
    __shared__ float maximum[kThreads];
    __shared__ unsigned bad[kThreads];
    float local = 0;
    unsigned nonfinite = 0;
    for (std::int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const float value = __bfloat162float(values[i]);
        if (!isfinite(value)) { nonfinite = 1; }
        else { local = fmaxf(local, fabsf(value)); }
    }
    maximum[threadIdx.x] = local;
    bad[threadIdx.x] = nonfinite;
    __syncthreads();
    for (int stride = kThreads / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) {
            maximum[threadIdx.x] = fmaxf(maximum[threadIdx.x], maximum[threadIdx.x + stride]);
            bad[threadIdx.x] |= bad[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        maxima[blockIdx.x] = maximum[0];
        invalid[blockIdx.x] = bad[0];
    }
}

struct Stat {
    float absmax = 0;
    std::uint64_t tokens = 0;
    std::uint64_t calls = 0;
    int width = 0;
    std::uint64_t samples_seen = 0;
    std::vector<Sample> samples;
};

class Collector final : public Gemma4MlpInputSink {
public:
    explicit Collector(std::size_t layers) : stats(layers * 2),
        device(kBlocks * (sizeof(float) + sizeof(unsigned))),
        host(kBlocks * (sizeof(float) + sizeof(unsigned))),
        sample_device(kSamplesPerCall * sizeof(Sample)),
        sample_host(kSamplesPerCall * sizeof(Sample)) {
        for (auto& stat : stats) { stat.samples.reserve(kSampleCapacity); }
    }

    void observe(std::size_t layer, Gemma4MlpInput input, const Tensor& value,
                 cudaStream_t stream) override {
        if (value.dtype != DType::BF16 || !value.is_contiguous() ||
            value.ne[2] != 1 || value.ne[3] != 1 || value.ne[0] <= 0 || value.ne[1] <= 0 ||
            value.ne[0] % 16 != 0) {
            throw std::runtime_error("calibration tap requires contiguous BF16 [K,T]");
        }
        auto* maxima = static_cast<float*>(device.p);
        auto* invalid = reinterpret_cast<unsigned*>(maxima + kBlocks);
        input_absmax<<<kBlocks, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(value.data), value.numel(), maxima, invalid);
        CUDA_CHECK(cudaGetLastError());
        auto& stat = stats.at(layer * 2 + (input == Gemma4MlpInput::Down ? 1 : 0));
        sample_inputs<<<1, 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(value.data), value.numel() / 16, stat.calls,
            static_cast<Sample*>(sample_device.p));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(sample_host.data(), sample_device.p, sample_device.bytes,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(host.data(), device.p, device.bytes,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto* max_host = static_cast<const float*>(host.data());
        const auto* bad_host = reinterpret_cast<const unsigned*>(max_host + kBlocks);
        for (int block = 0; block < kBlocks; ++block) {
            if (bad_host[block]) { throw std::runtime_error("nonfinite logical MLP input"); }
            stat.absmax = std::max(stat.absmax, max_host[block]);
        }
        const auto* samples = static_cast<const Sample*>(sample_host.data());
        const auto count = std::min<std::int64_t>(kSamplesPerCall, value.numel() / 16);
        for (int i = 0; i < count; ++i) {
            ++stat.samples_seen;
            if (stat.samples.size() < kSampleCapacity) { stat.samples.push_back(samples[i]); }
            else {
                const auto slot = mix(stat.samples_seen) % stat.samples_seen;
                if (slot < kSampleCapacity) { stat.samples[slot] = samples[i]; }
            }
        }
        if (stat.width && stat.width != value.ne[0]) {
            throw std::runtime_error("logical MLP input width changed");
        }
        stat.width = value.ne[0];
        stat.tokens += value.ne[1];
        ++stat.calls;
    }

    std::vector<Stat> stats;
private:
    DeviceBuffer device;
    PinnedHostBuffer host;
    DeviceBuffer sample_device;
    PinnedHostBuffer sample_host;
};

int collect(const char* artifact, const char* input, const char* output) {
    if (std::filesystem::exists(output)) { throw std::runtime_error("output already exists"); }
    std::ifstream file(input);
    if (!file) { throw std::runtime_error("cannot open calibration token JSON"); }
    Json rows;
    file >> rows;
    if (!rows.is_array() || rows.empty()) { throw std::runtime_error("empty calibration rows"); }
    const int context = static_cast<int>(rows.at(0).size());
    if (context < 1 || context > 16384) { throw std::runtime_error("context must be 1..16384"); }
    DeviceContext device;
    auto model = load_text(artifact, device, false);
    for (const auto& layer : model.layers) {
        for (auto policy : {layer.qkv_policy, layer.o_policy, layer.gate_up_policy, layer.down_policy}) {
            if (policy != ops::LinearPolicy::A16Only) {
                throw std::runtime_error("collector requires an entirely A16Only target trunk");
            }
        }
    }
    Collector collector(model.layers.size());
    const int pages = (context + 63) / 64;
    std::vector<std::int32_t> page_ids(pages), positions(context);
    std::iota(page_ids.begin(), page_ids.end(), 0);
    std::iota(positions.begin(), positions.end(), 0);
    DeviceBuffer table(pages * sizeof(std::int32_t));
    table.copy_from_host(page_ids.data(), table.bytes);
    Tensor block(table.p, DType::I32, {pages});
    std::vector<DeviceBuffer> keys, values;
    std::vector<Tensor> kt, vt;
    std::vector<Gemma4CacheLayer> caches;
    keys.reserve(model.layers.size()); values.reserve(model.layers.size());
    kt.reserve(model.layers.size()); vt.reserve(model.layers.size());
    caches.reserve(model.layers.size());
    std::size_t workspace_bytes = 0;
    for (const auto& layer : model.layers) {
        const std::size_t bytes = static_cast<std::size_t>(layer.head_dim) * 64 * layer.kv_heads * pages * 2;
        keys.emplace_back(bytes); values.emplace_back(bytes);
        kt.emplace_back(keys.back().p, DType::BF16, std::initializer_list<std::int32_t>{layer.head_dim, 64, layer.kv_heads, pages});
        vt.emplace_back(values.back().p, DType::BF16, std::initializer_list<std::int32_t>{layer.head_dim, 64, layer.kv_heads, pages});
        caches.push_back({.k_pages = &kt.back(), .v_pages = &vt.back(), .block_table = &block});
        workspace_bytes = std::max(workspace_bytes, gemma4_layer_workspace_bytes(layer, context));
    }
    DeviceArena workspace(workspace_bytes);
    DeviceBuffer ids(context * sizeof(std::int32_t)), pos(context * sizeof(std::int32_t));
    DeviceBuffer hidden(static_cast<std::size_t>(model.hidden) * context * 2);
    pos.copy_from_host(positions.data(), pos.bytes);
    Tensor id_tensor(ids.p, DType::I32, {context});
    Tensor pos_tensor(pos.p, DType::I32, {context});
    Tensor x(hidden.p, DType::BF16, {model.hidden, context});
    ops::Bf16GemmContext gemm(device.stream);
    std::vector<std::int32_t> tokens;
    tokens.reserve(context);
    for (std::size_t row = 0; row < rows.size(); ++row) {
        if (!rows[row].is_array() || rows[row].size() != static_cast<std::size_t>(context)) {
            throw std::runtime_error("calibration rows must have identical lengths");
        }
        tokens.clear();
        for (const auto& token : rows[row]) {
            if (!token.is_number_integer() || token.get<std::int64_t>() < 0 ||
                token.get<std::int64_t>() >= model.vocab) {
                throw std::runtime_error("calibration token outside vocabulary");
            }
            tokens.push_back(token.get<std::int32_t>());
        }
        // Each row is a separate causal sequence. Every readable position is overwritten by
        // qkv_prep before attention, with positions reset to zero and no prefix-cache reuse.
        ids.copy_from_host(tokens.data(), ids.bytes);
        workspace.reset();
        gemma4_layers(id_tensor, x, model.token_embedding, model.layers, caches, pos_tensor,
                       gemm, workspace, device.stream, nullptr, nullptr, &collector);
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        std::cerr << "calibration " << row + 1 << '/' << rows.size() << '\n';
    }
    Json statistics = Json::object();
    for (std::size_t i = 0; i < collector.stats.size(); ++i) {
        const auto& stat = collector.stats[i];
        if (stat.calls != rows.size() || stat.tokens != rows.size() * context) {
            throw std::runtime_error("incomplete logical input coverage");
        }
        const auto name = "text/layers/" + std::to_string(i / 2) + "/mlp/" +
                          (i % 2 ? "down" : "gate_up");
        Json samples = Json::array();
        for (const auto& sample : stat.samples) {
            Json probes = Json::array();
            for (std::size_t p = 0; p < kProbeDivisors.size(); ++p) {
                probes.push_back({{"scale", sample.scales[p]}, {"codes", {sample.lo[p], sample.hi[p]}}});
            }
            samples.push_back({{"call", sample.call}, {"block", sample.block},
                               {"values", Json::array_t(sample.values, sample.values + 16)},
                               {"probes", probes}});
        }
        statistics[name] = {{"absmax", stat.absmax}, {"tokens", stat.tokens},
                            {"calls", stat.calls}, {"width", stat.width},
                            {"samples_seen", stat.samples_seen}, {"samples", samples}};
    }
    Json result = {{"schema", "ninfer.gemma4.mlp-input-k16/v2"},
                   {"rows", rows.size()}, {"context", context}, {"kv_dtype", "bf16"},
                   {"qk", "bf16"}, {"statistics", statistics},
                   {"sampling", {{"method", "stratified-k16-reservoir-splitmix64/v1"},
                                 {"block_size", 16}, {"blocks_per_call", kSamplesPerCall},
                                 {"capacity", kSampleCapacity}, {"probe_divisors", kProbeDivisors}}}};
    std::ofstream out(output);
    if (!out) { throw std::runtime_error("cannot create statistics output"); }
    out << result.dump(2) << '\n';
    out.close();
    if (!out) { throw std::runtime_error("statistics write failed"); }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            throw std::runtime_error("usage: ninfer-spo-gemma4-calibrate ARTIFACT TOKENS.json STATS.json (use tools.convert.gemma4_a4 collect for identity-bound receipts)");
        }
        return collect(argv[1], argv[2], argv[3]);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
