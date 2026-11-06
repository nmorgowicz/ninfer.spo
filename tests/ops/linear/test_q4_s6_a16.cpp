// q4_g32_s6_fp16 A16 linear: the byte contract (golden block, production decode against the fixture's
// independent decode), the geometry contract, numerics against the FP64 dequantized-weight oracle on
// the Gemma 4 31B trunk shapes and head, bit-exact equality with the q4_g32 kernels on the same
// weights expressed as a g32 tensor (every group scale = binary16 round to nearest even of S * m_j),
// and a cold-L2 bandwidth bench (NINFER_Q4_COLD_S6=<T>). W4A8 is test_q4_s6_a8.cpp.
#include "ops/linear/linear_test_common.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/weight_view.h"
#include "ninfer/ops/linear.h"
#include "ops/common/q4_scale_mode.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer::test::linear;
using ninfer::QType;
namespace qw  = ninfer::test::quantized_weight;
namespace det = ninfer::ops::detail;

constexpr Invocation a16(std::int32_t t) { return {t}; }

constexpr std::array kTrunkN{16384, 18432, 43008, 5376};
constexpr std::array kTrunkK{5376, 8192, 16384, 21504};

// T=1 (unrolled DRAM GEMV), 2..8 and 9..16 (tensor-core passes), 17 and 33 (wide).
constexpr std::array kTokens{a16(1),  a16(2),  a16(3),  a16(4),  a16(5),  a16(6),
                             a16(7),  a16(8),  a16(9),  a16(12), a16(16), a16(17),
                             a16(33)};
// Large prefill-like token counts on the wide kernel.
constexpr std::array kWideTokens{a16(64), a16(129), a16(1024)};
// Shapes the tensor-core and unrolled T=1 kernels reject land on the shared-tile GEMV (T=1..8 and 16)
// and the wide kernel (9, 17, 33).
constexpr std::array kFallbackTokens{a16(1), a16(2), a16(3), a16(4),  a16(5),  a16(6), a16(7),
                                     a16(8), a16(9), a16(16), a16(17), a16(33)};

template <typename Exception, typename Call>
bool throws(Call&& call) {
    try {
        call();
    } catch (const Exception&) {
        return true;
    }
    return false;
}

// The byte contract, once against literal expected bits, then against the production helper for
// every finite binary16 super scale and every multiplier (the fixture decode is the oracle).
int decode_contract() {
    int failures = 0;
    constexpr std::array<std::uint8_t, 8> kBlock{0x55, 0x35, 0x40, 0x20, 0x0c, 0xbf, 0xff, 0x45};
    constexpr std::array<std::uint16_t, 8> kBits{0x0000, 0x3555, 0x3955, 0x3c00,
                                                 0x4d40, 0x4d2a, 0x492a, 0x45aa};
    const std::vector<std::uint8_t> bytes(kBlock.begin(), kBlock.end());
    const std::uint64_t fixture_word = qw::detail::load_u64_le(bytes, 0);
    const std::uint64_t word         = det::q4_s6_block_word(kBlock.data());
    if (word != 0x45ffbf0c20403555ULL || fixture_word != word) {
        std::cerr << "S6 golden block: little-endian word is 0x" << std::hex << word << " (fixture 0x"
                  << fixture_word << ")" << std::dec << '\n';
        ++failures;
    }
    for (int j = 0; j < 8; ++j) {
        const std::uint16_t production = det::q4_s6_scale_bits(word, j);
        const std::uint16_t oracle     = qw::detail::s6_group_scale_bits(word, j);
        if (production != kBits[static_cast<std::size_t>(j)] ||
            oracle != kBits[static_cast<std::size_t>(j)]) {
            std::cerr << "S6 golden block group " << j << ": production 0x" << std::hex << production
                      << " fixture 0x" << oracle << " expected 0x" << kBits[static_cast<std::size_t>(j)]
                      << std::dec << '\n';
            ++failures;
        }
        if (det::q4_s6_scale(word, j) != qw::detail::f16_to_f32(kBits[static_cast<std::size_t>(j)])) {
            std::cerr << "S6 golden block group " << j << ": fp32 scale differs from the binary16 value\n";
            ++failures;
        }
    }
    // Round trip of the fixture's writer: the same block packed from S and m.
    const std::uint8_t multipliers[8]{0, 1, 2, 3, 63, 62, 31, 17};
    if (qw::detail::s6_pack_block(0x3555, multipliers) != word) {
        std::cerr << "S6 golden block: fixture packing differs from the contract bytes\n";
        ++failures;
    }

    std::size_t mismatches = 0;
    for (std::uint32_t super = 0; super < 0x10000U; ++super) {
        if (((super >> 10) & 31U) == 31U) { continue; }
        for (std::uint32_t base = 0; base < 64U; ++base) {
            std::uint8_t m[8];
            for (std::uint32_t j = 0; j < 8U; ++j) { m[j] = static_cast<std::uint8_t>((base + 8U * j) & 63U); }
            const std::uint64_t block = qw::detail::s6_pack_block(static_cast<std::uint16_t>(super), m);
            for (int j = 0; j < 8; ++j) {
                if (det::q4_s6_scale_bits(block, j) != qw::detail::s6_group_scale_bits(block, j)) {
                    if (mismatches++ < 4) {
                        std::cerr << "S6 decode: S=0x" << std::hex << super << " m=" << std::dec
                                  << static_cast<int>(m[j]) << ": production and fixture differ\n";
                    }
                }
            }
        }
    }
    if (mismatches != 0) {
        std::cerr << "S6 decode: " << mismatches << " production/fixture mismatches\n";
        ++failures;
    }
    return failures;
}

// The fixture's `dequant`, its independent logical decode and the g32 emulation agree, and a row view
// of the S6 payload advances the scale plane by K / 32 bytes per row.
int fixture_contract() {
    int failures = 0;
    for (const auto [n, k] : {std::pair{16, 512}, std::pair{9, 250}, std::pair{5, 256}}) {
        auto packed          = qw::make_q4_g32_s6_weight(n, k, 17U, true);
        const auto emulated  = qw::emulate_q4_g32_from_s6(packed);
        const std::size_t row_bytes = static_cast<std::size_t>(packed.weight.padded_shape[1]) / 32;
        if (packed.scale_plane_bytes != static_cast<std::uint64_t>(n) * row_bytes ||
            packed.scale_plane_offset % 256 != 0 || packed.high_plane_bytes != 0 ||
            packed.weight.scale_dtype != ninfer::DType::U8 || packed.weight.group != 32 ||
            packed.weight.group_size != 32) {
            std::cerr << "S6 fixture [" << n << "," << k << "]: plane or metadata layout is wrong\n";
            ++failures;
        }
        std::size_t bad = 0;
        for (std::int32_t row = 0; row < n; ++row) {
            for (std::int32_t column = 0; column < k; ++column) {
                const double expected = packed.dequant[static_cast<std::size_t>(row) * k + column];
                bad += qw::logical_weight_fp64(packed, row, column) != expected ||
                       qw::logical_weight_fp64(emulated, row, column) != expected;
            }
        }
        if (bad != 0) {
            std::cerr << "S6 fixture [" << n << "," << k << "]: " << bad
                      << " elements differ between dequant, logical decode and the g32 emulation\n";
            ++failures;
        }
        const ninfer::Weight view = packed.device_row_view(packed.payload.data(), 3, 2);
        if (view.n != 2 ||
            static_cast<const std::uint8_t*>(view.scales) !=
                packed.payload.data() + packed.scale_plane_offset + 3 * row_bytes ||
            static_cast<const std::uint8_t*>(view.qdata) !=
                packed.payload.data() + 3 * static_cast<std::size_t>(packed.weight.padded_shape[1]) / 2) {
            std::cerr << "S6 fixture [" << n << "," << k << "]: row view planes are wrong\n";
            ++failures;
        }
    }
    // Random blocks cover the extremes the contract names: a zero and a 63 multiplier.
    const auto packed = qw::make_q4_g32_s6_weight(64, 2048, 5U);
    bool saw_zero = false, saw_max = false;
    for (std::size_t i = 0; i < packed.scale_plane_bytes / 8; ++i) {
        const std::uint64_t word =
            qw::detail::load_u64_le(packed.payload, static_cast<std::size_t>(packed.scale_plane_offset) + i * 8);
        for (int j = 0; j < 8; ++j) {
            const std::uint64_t m = (word >> (16 + 6 * j)) & 63U;
            saw_zero |= m == 0;
            saw_max |= m == 63;
        }
    }
    if (!saw_zero || !saw_max) {
        std::cerr << "S6 fixture: multipliers 0 and 63 are not both present\n";
        ++failures;
    }
    if (!throws<std::invalid_argument>([] { (void)qw::make_q4_g32_s6_weight(8, 5504, 1U); })) {
        std::cerr << "S6 fixture accepted K=5504\n";
        ++failures;
    }
    return failures;
}

int numerics() {
    int failures       = 0;
    std::uint32_t seed = 101U;
    for (const std::int32_t n : kTrunkN) {
        for (const std::int32_t k : kTrunkK) {
            failures += run_shape("Q4_G32_S6", ActivationCompute::A16, make_q4_g32_s6_fp16_weight,
                                  {n, k, seed++, Comparison::Sampled, true, kTokens});
        }
    }
    failures += run_shape("Q4_G32_S6", ActivationCompute::A16, make_q4_g32_s6_fp16_weight,
                          {16384, 5376, seed++, Comparison::Sampled, true, kWideTokens});
    failures += run_shape("Q4_G32_S6", ActivationCompute::A16, make_q4_g32_s6_fp16_weight,
                          {5376, 21504, seed++, Comparison::Sampled, true, kWideTokens});
    // The head: one 8-row-per-warp T=1 pass, and 64-row tensor-core tiles.
    constexpr std::array kHeadCalls{a16(1), a16(4)};
    failures += run_shape("Q4_G32_S6", ActivationCompute::A16, make_q4_g32_s6_fp16_weight,
                          {262144, 5376, seed++, Comparison::Sampled, true, kHeadCalls});
    // Rows or K the fast kernels do not tile: every row and token is compared.
    failures += run_shape("Q4_G32_S6", ActivationCompute::A16, make_q4_g32_s6_fp16_weight,
                          {136, 5376, seed++, Comparison::Full, true, kFallbackTokens});
    failures += run_shape("Q4_G32_S6", ActivationCompute::A16, make_q4_g32_s6_fp16_weight,
                          {256, 1792, seed++, Comparison::Full, true, kFallbackTokens});
    failures += run_shape("Q4_G32_S6", ActivationCompute::A16, make_q4_g32_s6_fp16_weight,
                          {128, 256, seed++, Comparison::Full, true, kTokens});
    failures += run_shape("Q4_G32_S6", ActivationCompute::A16, make_q4_g32_s6_fp16_weight,
                          {72, 768, seed++, Comparison::Full, true, kFallbackTokens});
    // 168-byte scale rows: no 16-byte alignment per row.
    failures += run_shape("Q4_G32_S6", ActivationCompute::A16, make_q4_g32_s6_fp16_weight,
                          {33, 5376, seed++, Comparison::Full, true, kFallbackTokens});
    return failures;
}

std::vector<std::uint16_t> make_bf16_activation(std::int32_t k, std::int32_t t, std::uint32_t seed) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(k) * t);
    std::uint32_t state = 0x9e3779b9u ^ seed;
    for (auto& word : bits) {
        state             = state * 1664525u + 1013904223u;
        const float value = (static_cast<float>((state >> 8) & 0xffffu) / 32768.0f - 1.0f) * 2.0f;
        word              = ninfer::test::f32_to_bf16(value);
    }
    return bits;
}

// Same decoded weights, two layouts: the S6 kernels must reproduce the g32 kernels' bits.
int equals_emulated_g32(std::int32_t n, std::int32_t k, std::span<const Invocation> calls,
                        std::uint32_t seed) {
    using namespace ninfer;
    const auto native   = make_q4_g32_s6_fp16_weight(n, k, seed);
    const auto emulated = qw::emulate_q4_g32_from_s6(native);
    DeviceBuffer native_device(native.payload.size());
    native_device.copy_from_host(native.payload.data(), native_device.bytes);
    DeviceBuffer emulated_device(emulated.payload.size());
    emulated_device.copy_from_host(emulated.payload.data(), emulated_device.bytes);
    const Weight native_weight   = native.device_weight(native_device.p);
    const Weight emulated_weight = emulated.device_weight(emulated_device.p);

    const std::int32_t max_t =
        std::max_element(calls.begin(), calls.end(), [](const Invocation& a, const Invocation& b) {
            return a.t < b.t;
        })->t;
    const auto activation = make_bf16_activation(k, max_t, seed + 7U);
    DeviceBuffer device_activation(activation.size() * 2);
    device_activation.copy_from_host(activation.data(), device_activation.bytes);

    int failures = 0;
    for (const Invocation& call : calls) {
        const std::string label = "Q4_G32_S6 == emulated g32 [" + std::to_string(n) + "," +
                                  std::to_string(k) + "] T=" + std::to_string(call.t);
        const std::size_t words = static_cast<std::size_t>(n) * call.t;
        DeviceBuffer native_out(words * 2), emulated_out(words * 2);
        std::vector<std::uint8_t> poison(words * 2, 0xff);
        native_out.copy_from_host(poison.data(), poison.size());
        emulated_out.copy_from_host(poison.data(), poison.size());
        Tensor input(device_activation.p, DType::BF16, {k, call.t});
        Tensor native_dst(native_out.p, DType::BF16, {n, call.t});
        Tensor emulated_dst(emulated_out.p, DType::BF16, {n, call.t});
        try {
            ops::linear(input, native_weight, native_dst, nullptr);
            ops::linear(input, emulated_weight, emulated_dst, nullptr);
            ninfer::test::cuda_check(cudaDeviceSynchronize(), "synchronize S6 equivalence");
        } catch (const std::exception& error) {
            std::cerr << label << ": unexpected exception: " << error.what() << '\n';
            ++failures;
            continue;
        }
        std::vector<std::uint16_t> got(words), want(words);
        native_out.copy_to_host(got.data(), got.size() * 2);
        emulated_out.copy_to_host(want.data(), want.size() * 2);
        if (got != want) {
            std::size_t first = 0;
            while (got[first] == want[first]) { ++first; }
            std::cerr << label << ": outputs differ from the g32 kernel on the emulated tensor, first at row "
                      << first % n << " token " << first / n << " (0x" << std::hex << got[first]
                      << " vs 0x" << want[first] << std::dec << ")\n";
            ++failures;
        }
        if (std::count(got.begin(), got.end(), std::uint16_t{0xffff}) != 0 ||
            std::count(got.begin(), got.end(), std::uint16_t{0}) == static_cast<std::ptrdiff_t>(got.size())) {
            std::cerr << label << ": output is poisoned or all zero\n";
            ++failures;
        }
    }
    return failures;
}

int equivalence() {
    int failures       = 0;
    std::uint32_t seed = 900U;
    constexpr std::array kBitTokens{a16(1), a16(2), a16(5), a16(7), a16(8), a16(9),
                                    a16(16), a16(17), a16(40)};
    for (const std::int32_t n : kTrunkN) {
        for (const std::int32_t k : kTrunkK) {
            failures += equals_emulated_g32(n, k, kBitTokens, seed++);
        }
    }
    failures += equals_emulated_g32(262144, 5376, std::array{a16(1), a16(4)}, seed++);
    failures += equals_emulated_g32(136, 5376, kFallbackTokens, seed++);
    failures += equals_emulated_g32(256, 1792, kFallbackTokens, seed++);
    failures += equals_emulated_g32(5376 + 16, 5376, kBitTokens, seed++);
    failures += equals_emulated_g32(72, 768, kFallbackTokens, seed++);
    return failures;
}

// The runtime's geometry (artifact sizes) must match the fixture byte for byte, and K whose padded
// value is not a multiple of 256 must be refused instead of floored.
int geometry_contract() {
    using namespace ninfer;
    int failures = 0;
    for (const std::uint64_t k : {std::uint64_t{256}, std::uint64_t{1792}, std::uint64_t{5376},
                                  std::uint64_t{8192}, std::uint64_t{21504}}) {
        const std::array<std::uint64_t, 2> shape{48, k};
        const WeightGeometry geometry =
            weight_geometry(QType::Q4_G32_S6_FP16, QuantLayout::RowSplit, shape);
        const auto packed = make_q4_g32_s6_fp16_weight(48, static_cast<std::int32_t>(k), 3U);
        if (geometry.group_size != 32 || geometry.code_bytes_per_row != k / 2 ||
            geometry.high_bytes_per_row != 0 || geometry.scale_bytes_per_row != k / 32 ||
            geometry.scale_bytes != 48 * (k / 32) || geometry.padded_columns != k ||
            geometry.bytes != packed.payload.size() || geometry.scale_offset != packed.scale_plane_offset ||
            geometry.scale_offset % 256 != 0) {
            std::cerr << "Q4_G32_S6 K=" << k << ": geometry disagrees with the fixture\n";
            ++failures;
        }
        for (const std::uint64_t row : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{47}}) {
            for (const std::uint64_t group : {std::uint64_t{0}, std::uint64_t{7}, std::uint64_t{8},
                                              k / 32 - 1}) {
                if (group >= k / 32) { continue; }
                const std::uint64_t expected =
                    packed.scale_plane_offset + (row * (k / 256) + group / 8) * 8;
                if (weight_scale_offset(geometry, row, group) != expected) {
                    std::cerr << "Q4_G32_S6 K=" << k << ": scale block offset of row " << row << " group "
                              << group << " is wrong\n";
                    ++failures;
                }
            }
        }
    }
    // K = 250 pads to 256: accepted; its row is one block.
    {
        const std::array<std::uint64_t, 2> shape{4, 250};
        const WeightGeometry geometry =
            weight_geometry(QType::Q4_G32_S6_FP16, QuantLayout::RowSplit, shape);
        const auto packed = make_q4_g32_s6_fp16_weight(4, 250, 3U);
        if (geometry.padded_columns != 256 || geometry.scale_bytes_per_row != 8 ||
            geometry.bytes != packed.payload.size()) {
            std::cerr << "Q4_G32_S6 K=250: geometry disagrees with the fixture\n";
            ++failures;
        }
    }
    // Padded K that is not a multiple of 256 is refused (g32 would pad it; a block cannot).
    for (const std::uint64_t k : {std::uint64_t{128}, std::uint64_t{300}, std::uint64_t{640},
                                  std::uint64_t{5248}, std::uint64_t{5440}, std::uint64_t{5504}}) {
        const std::array<std::uint64_t, 2> shape{16, k};
        if (!throws<std::invalid_argument>(
                [&] { (void)weight_geometry(QType::Q4_G32_S6_FP16, QuantLayout::RowSplit, shape); })) {
            std::cerr << "Q4_G32_S6: K=" << k << " (padded K not a multiple of 256) was accepted\n";
            ++failures;
        }
    }
    // The single-level formats keep accepting K = 5504 and keep their scale sizes.
    {
        const std::array<std::uint64_t, 2> shape{16, 5504};
        const WeightGeometry g32 = weight_geometry(QType::Q4_G32_FP16, QuantLayout::RowSplit, shape);
        if (g32.scale_bytes_per_row != 5504 / 16) {
            std::cerr << "Q4_G32: scale size changed\n";
            ++failures;
        }
    }
    return failures;
}

void bench_cold(const char* name, qw::PackedWeight packed, double scale_bytes_per_weight, std::int32_t n,
                std::int32_t k, std::int32_t t) {
    using namespace ninfer;
    const std::size_t bytes = packed.payload.size();
    const int copies        = static_cast<int>(std::max<std::size_t>(2, (std::size_t{512} << 20) / bytes + 1));
    std::vector<DeviceBuffer> buffers;
    std::vector<Weight> weights;
    buffers.reserve(copies);
    for (int c = 0; c < copies; ++c) {
        buffers.emplace_back(bytes);
        buffers.back().copy_from_host(packed.payload.data(), bytes);
        weights.push_back(packed.device_weight(buffers.back().p));
    }
    std::vector<std::uint16_t> activation(static_cast<std::size_t>(k) * t, 0x3f80);
    DeviceBuffer act(activation.size() * 2);
    act.copy_from_host(activation.data(), act.bytes);
    DeviceBuffer out(static_cast<std::size_t>(n) * t * 2);
    Tensor input(act.p, DType::BF16, {k, t});
    Tensor destination(out.p, DType::BF16, {n, t});
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for (const Weight& weight : weights) { ops::linear(input, weight, destination, stream); }
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    const int warm = std::max(20, 4000 / copies);
    for (int i = 0; i < warm; ++i) { cudaGraphLaunch(exec, stream); }
    cudaStreamSynchronize(stream);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    const int iters = std::max(10, 400 / copies);
    cudaEventRecord(start, stream);
    for (int i = 0; i < iters; ++i) { cudaGraphLaunch(exec, stream); }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    const double seconds      = (ms / 1000.0) / (static_cast<double>(iters) * copies);
    const double weight_bytes = static_cast<double>(n) * k * (0.5 + scale_bytes_per_weight);
    std::cout << "cold " << name << " " << n << "x" << k << " T=" << t << " " << (seconds * 1e6) << " us "
              << (weight_bytes / seconds / 1e9) << " GB/s (" << copies << " copies)\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
}

} // namespace

int main() {
    try {
        if (const char* cold = std::getenv("NINFER_Q4_COLD_S6")) {
            const int t = std::max(1, std::atoi(cold));
            for (auto [n, k] : {std::pair{16384, 5376}, std::pair{18432, 5376}, std::pair{43008, 5376},
                                std::pair{5376, 8192}, std::pair{5376, 16384}, std::pair{5376, 21504},
                                std::pair{262144, 5376}}) {
                bench_cold("Q4_G32   ", make_q4_g32_fp16_weight(n, k, 11U), 1.0 / 16.0, n, k, t);
                bench_cold("Q4_G32_S6", make_q4_g32_s6_fp16_weight(n, k, 11U), 1.0 / 32.0, n, k, t);
            }
            return 0;
        }
        int failures = decode_contract();
        failures += fixture_contract();
        failures += geometry_contract();
        if (!cuda_available()) {
            std::cerr << "Q4 S6 test: CUDA is required\n";
            return 1;
        }
        failures += numerics();
        failures += equivalence();
        if (failures != 0) {
            std::cerr << "Q4 S6 test: " << failures << " failures\n";
            return 1;
        }
        std::cout << "OK Q4 S6 A16\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Q4 S6 test: " << error.what() << '\n';
        return 1;
    }
}
