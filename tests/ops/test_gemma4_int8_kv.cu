// INT8-G64 global-layer KV cache for Gemma4 (D512, 8 FP16 scales per vector, SIMT/MMA
// dequant-to-BF16 reads, no rotation). Mirrors test_gemma4_fp8_kv.cu without the FP8 QK path.
#include "core/runtime_options.h"
#include "ninfer/ops/gemma4_attention.h"
#include "ninfer/ops/gemma4_kv_append.h"
#include "ninfer/ops/gemma4_qkv_prep.h"
#include "ops/kernel/gemma4_int8_kv.cuh"
#include "ops/op_tester.h"
#include "ops/wrapper/gemma4_cache_validation.h"

#include <cmath>
#include <limits>

using namespace ninfer;
using namespace ninfer::test;

namespace {
constexpr int D = 512;
constexpr int G = 8; // INT8 groups per vector
constexpr int Heads = 2;
constexpr int Pages = 4;
constexpr int Rows = 64 * Heads * Pages;

std::uint16_t half_bits(float x) {
    const __half h = __float2half_rn(x);
    return reinterpret_cast<const __half_raw&>(h).x;
}
float half_value(std::uint16_t bits) {
    __half_raw raw;
    raw.x = bits;
    return __half2float(__half(raw));
}

// Host statement of the numerical contract: scale = FP16-RNE(absmax/127), codes use the
// reciprocal of the represented scale, round-to-nearest-even, clamp +-127. A group with any
// nonfinite value (or an unrepresentable scale) reads back as NaN: NaN scale, zero codes.
struct Group {
    bool poison = false;
    std::uint16_t scale = 0;
};
Group quantize_group(const std::uint16_t* x, std::int8_t* codes) {
    float maximum = 0.0f;
    Group group;
    for (int i = 0; i < 64; ++i) {
        const float v = bf16_to_f32(x[i]);
        group.poison |= !std::isfinite(v);
        maximum = std::fmax(maximum, std::fabs(v));
    }
    group.scale = half_bits(maximum > 0.0f ? maximum / 127.0f : 0.0f);
    const float represented = half_value(group.scale);
    group.poison |= !std::isfinite(represented);
    const float inverse = represented > 0.0f ? 1.0f / represented : 0.0f;
    for (int i = 0; i < 64; ++i) {
        if (group.poison || inverse == 0.0f) { codes[i] = 0; continue; }
        const float q = std::nearbyint(bf16_to_f32(x[i]) * inverse);
        codes[i] = static_cast<std::int8_t>(std::max(-127.0f, std::min(127.0f, q)));
    }
    if (group.poison) { group.scale = 0x7e00; }
    return group;
}
// Row quantizer used where inputs are finite; scales are exact bits.
void quantize(const std::uint16_t* x, std::uint8_t* codes, std::uint16_t* scales) {
    for (int g = 0; g < G; ++g) {
        scales[g] = quantize_group(x + g * 64, reinterpret_cast<std::int8_t*>(codes) + g * 64).scale;
    }
}
float expand_ref(std::uint8_t code, std::uint16_t scale) {
    return static_cast<float>(static_cast<std::int8_t>(code)) * half_value(scale);
}

__global__ void codec_kernel(const __nv_bfloat162* x, std::int8_t* codes, __half* scales) {
    __nv_bfloat162 values[8];
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    for (int s = 0; s < 8; ++s) { values[s] = x[row * 256 + lane + 32 * s]; }
    ops::detail::gemma4_int8_store(values, codes, scales, row, lane);
}
__global__ void expand_kernel(const std::int8_t* codes, const __half* scales,
                              __nv_bfloat16* out, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { out[i] = ops::detail::gemma4_int8_expand(codes[i], scales[i / 64]); }
}

int codec_test() {
    constexpr int rows = 12;
    constexpr int count = rows * D;
    std::vector<float> values(count, 0.0f);
    for (int row = 1; row < rows; ++row) {
        for (int i = 0; i < D; ++i) {
            // Every group uses a different amplitude so group scales cannot be shared.
            values[row * D + i] = std::ldexp(1.0f, i / 64 - 3) * float((i * 19) % 101 - 50) / 37.0f;
        }
    }
    for (int i = D; i < 2 * D; ++i) { values[i] *= 0x1p-100f; }           // scale underflows to 0
    for (int i = 2 * D; i < 3 * D; ++i) { values[i] *= 0x1p10f; }          // large finite
    values[3 * D + 130] = std::numeric_limits<float>::quiet_NaN();        // group 2 only
    values[4 * D + 301] = std::numeric_limits<float>::infinity();         // group 4 only
    values[5 * D + 31] = -std::numeric_limits<float>::infinity();         // group 0 only
    for (int i = 6 * D; i < 7 * D; ++i) { values[i] = std::numeric_limits<float>::quiet_NaN(); }
    values[7 * D + 63] = 0.252f;                                           // group boundaries
    values[7 * D + 64] = -15.125f;
    values[8 * D + 0] = -0.0f;
    for (int i = 0; i < 64; ++i) { values[9 * D + i] = (i & 1 ? -1.0f : 1.0f) * 127.0f; } // +-saturation
    values[10 * D + 5] = 1.0e30f;                                          // scale overflows FP16
    for (int i = 0; i < 64; ++i) { values[11 * D + 448 + i] = float(i - 31) * 0.01f; }  // last group
    auto input = to_device_bf16(values);
    const auto input_bits = from_device<std::uint16_t>(input, count);
    GuardedDeviceBuffer codes(count), scales(rows * G * 2), expanded(count * 2);
    codec_kernel<<<rows, 32>>>(static_cast<const __nv_bfloat162*>(input.p),
        static_cast<std::int8_t*>(codes.data()), static_cast<__half*>(scales.data()));
    cuda_check_last_launch("int8 codec_kernel");
    expand_kernel<<<(count + 127) / 128, 128>>>(static_cast<const std::int8_t*>(codes.data()),
        static_cast<const __half*>(scales.data()), static_cast<__nv_bfloat16*>(expanded.data()), count);
    cuda_synchronize();
    const auto got_codes = from_device<std::uint8_t>(codes.data(), count);
    const auto got_scales = from_device<std::uint16_t>(scales.data(), rows * G);
    const auto got = from_device<std::uint16_t>(expanded.data(), count);
    int failures = 0;
    int poisoned_groups = 0;
    for (int row = 0; row < rows && !failures; ++row) {
        for (int g = 0; g < G; ++g) {
            std::int8_t expected_codes[64];
            const auto group = quantize_group(input_bits.data() + row * D + g * 64, expected_codes);
            const auto scale = got_scales[row * G + g];
            if (group.poison) {
                ++poisoned_groups;
                if (!std::isnan(half_value(scale))) {
                    std::cerr << "int8 poisoned group has finite scale row=" << row << " g=" << g << '\n';
                    ++failures; break;
                }
            } else if (scale != group.scale) {
                std::cerr << "int8 represented scale mismatch row=" << row << " g=" << g << '\n';
                ++failures; break;
            }
            for (int i = 0; i < 64; ++i) {
                const int index = row * D + g * 64 + i;
                if (static_cast<std::int8_t>(got_codes[index]) != expected_codes[i]) {
                    std::cerr << "int8 code mismatch at " << index << '\n';
                    ++failures; break;
                }
                const float x = bf16_to_f32(input_bits[index]);
                const float got_value = bf16_to_f32(got[index]);
                if (group.poison) {
                    if (!std::isnan(got_value)) {
                        std::cerr << "int8 poisoned group did not read back as NaN at " << index << '\n';
                        ++failures; break;
                    }
                    continue;
                }
                const float reference = expand_ref(got_codes[index], scale);
                if (got[index] != f32_to_bf16(reference)) {
                    std::cerr << "int8 FP32-product/BF16 expansion mismatch at " << index << '\n';
                    ++failures; break;
                }
                // Round-trip bound: half a code step, scale-rounding clip near saturation,
                // and the BF16 rounding of the represented value.
                const float s = half_value(scale);
                const float bound = 0.5f * s * 1.001f + std::fabs(x) * 0x1p-10f +
                                    std::fabs(reference) * 0x1p-8f;
                // A scale that underflows FP16 represents the group as zero by contract.
                if (s != 0.0f && !(std::fabs(got_value - x) <= bound)) {
                    std::cerr << "int8 round-trip error out of bound at " << index << " x=" << x
                              << " got=" << got_value << " scale=" << s << '\n';
                    ++failures; break;
                }
            }
            if (failures) { break; }
        }
    }
    // Exactly the groups containing NaN/Inf/overflow are poisoned: 1 + 1 + 1 + 8 + 1 (+ row 2's
    // huge-but-finite values must stay finite).
    if (poisoned_groups != 12) {
        std::cerr << "unexpected poisoned group count " << poisoned_groups << '\n';
        ++failures;
    }
    // Saturation row: +-127 map to +-127 codes with scale exactly 1.
    if (got_scales[9 * G] != half_bits(1.0f) || static_cast<std::int8_t>(got_codes[9 * D]) != 127 ||
        static_cast<std::int8_t>(got_codes[9 * D + 1]) != -127) {
        std::cerr << "int8 saturation row mismatch\n"; ++failures;
    }
    failures += codes.verify_guards("int8 codes") + scales.verify_guards("int8 scales") +
                expanded.verify_guards("int8 expanded");
    return failures;
}

int qkv_test() {
    const std::vector<int> positions{-1, 0, 1, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256};
    const std::vector<int> table{2, 0, -1, Pages};
    const int tokens = static_cast<int>(positions.size());
    std::vector<float> input((4 + Heads) * D * tokens), weights(D);
    fill_uniform(input, 101, -3.0f, 3.0f);
    fill_uniform(weights, 102, 0.4f, 1.7f);
    auto dx = to_device_bf16(input), dw = to_device_bf16(weights);
    auto dp = to_device_i32(positions), dt = to_device_i32(table);
    GuardedDeviceBuffer q(4 * D * tokens * 2), qr(q.bytes());
    GuardedDeviceBuffer k(Rows * D), v(Rows * D), ks(Rows * G * 2), vs(Rows * G * 2);
    GuardedDeviceBuffer kr(Rows * D * 2), vr(Rows * D * 2);
    std::vector<std::uint8_t> background(Rows * D, 0x5a);
    std::vector<std::uint16_t> scale_background(Rows * G, 0x3555), bf_background(Rows * D, 0x3f80);
    k.copy_from_host(background.data(), k.bytes()); v.copy_from_host(background.data(), v.bytes());
    ks.copy_from_host(scale_background.data(), ks.bytes()); vs.copy_from_host(scale_background.data(), vs.bytes());
    kr.copy_from_host(bf_background.data(), kr.bytes()); vr.copy_from_host(bf_background.data(), vr.bytes());
    Tensor x(dx.p, DType::BF16, {(4 + Heads) * D, tokens}), w(dw.p, DType::BF16, {D});
    Tensor p(dp.p, DType::I32, {tokens}), t(dt.p, DType::I32, {Pages});
    Tensor qo(q.data(), DType::BF16, {D, 4, tokens}), qro(qr.data(), DType::BF16, {D, 4, tokens});
    Tensor kt(k.data(), DType::I8, {D, 64, Heads, Pages}), vt(v.data(), DType::I8, {D, 64, Heads, Pages});
    Tensor kst(ks.data(), DType::FP16, {G, 64, Heads, Pages}), vst(vs.data(), DType::FP16, {G, 64, Heads, Pages});
    Tensor krt(kr.data(), DType::BF16, {D, 64, Heads, Pages}), vrt(vr.data(), DType::BF16, {D, 64, Heads, Pages});
    ops::Gemma4CacheView cache{ops::Gemma4KvFormat::Int8Group64, &kt, &vt, &kst, &vst, &t};
    ops::Gemma4CacheView reference{ops::Gemma4KvFormat::BFloat16, &krt, &vrt, nullptr, nullptr, &t};
    ops::gemma4_qkv_prep(x, w, w, p, 128, 1000000.0f, 1e-6f, true, qo, cache, nullptr);
    ops::gemma4_qkv_prep(x, w, w, p, 128, 1000000.0f, 1e-6f, true, qro, reference, nullptr);
    cuda_synchronize();
    int failures = verify_exact("int8 qkv Q unchanged", from_device<std::uint16_t>(q.data(), q.bytes() / 2),
                                 from_device<std::uint16_t>(qr.data(), qr.bytes() / 2));
    const auto kb = from_device<std::uint16_t>(kr.data(), Rows * D);
    const auto vb = from_device<std::uint16_t>(vr.data(), Rows * D);
    auto ek = background, ev = background;
    auto eks = scale_background, evs = scale_background;
    for (int pos : positions) {
        if (pos < 0 || pos / 64 >= Pages) { continue; }
        const int physical = table[pos / 64];
        if (physical < 0 || physical >= Pages) { continue; }
        for (int h = 0; h < Heads; ++h) {
            const auto row = ops::detail::gemma4_kv_row(physical, Heads, h, pos % 64);
            quantize(kb.data() + row * D, ek.data() + row * D, eks.data() + row * G);
            quantize(vb.data() + row * D, ev.data() + row * D, evs.data() + row * G);
        }
    }
    failures += verify_exact("int8 qkv K codes after norm/rope", from_device<std::uint8_t>(k.data(), ek.size()), ek);
    failures += verify_exact("int8 qkv V codes from raw K", from_device<std::uint8_t>(v.data(), ev.size()), ev);
    failures += verify_exact("int8 qkv K scales", from_device<std::uint16_t>(ks.data(), eks.size()), eks);
    failures += verify_exact("int8 qkv V scales", from_device<std::uint16_t>(vs.data(), evs.size()), evs);
    for (auto* buffer : {&q, &k, &v, &ks, &vs}) { failures += buffer->verify_guards("qkv INT8"); }
    return failures;
}

struct CacheFixture {
    std::vector<std::uint8_t> codes;
    std::vector<std::uint16_t> scales, expanded;
};

// Attention over arbitrary-but-valid codes: the INT8 paths must consume exactly the BF16
// values `expanded`, so the BF16 reference over `expanded` is bit-identical. The unwritten
// tail of the last page carries NaN scales and nonzero codes that must never be read.
int attention_test(int tokens, bool invalid_high_page = true, bool split_context = false,
                   bool empty = false) {
    std::vector<int> table{2, -1, 0, invalid_high_page ? Pages : 1};
    if (split_context) { table.resize(16, -1); }
    if (empty) { std::fill(table.begin(), table.end(), -1); }
    std::vector<int> positions(tokens);
    for (int i = 0; i < tokens; ++i) { positions[i] = 65 + i; }
    std::vector<float> query(D * 4 * tokens);
    fill_uniform(query, 301, -0.05f, 0.05f);
    auto dq = to_device_bf16(query), dp = to_device_i32(positions), dt = to_device_i32(table);
    CacheFixture cache_data;
    cache_data.codes.resize(Rows * D);
    cache_data.scales.resize(Rows * G);
    cache_data.expanded.resize(Rows * D);
    for (int row = 0; row < Rows; ++row) {
        const int last = positions.back();
        const bool unwritten = table[last / 64] == row / (64 * Heads) && row % 64 > last % 64;
        for (int g = 0; g < G; ++g) {
            cache_data.scales[row * G + g] = unwritten ? 0x7e00
                : half_bits(0.0011f + ((row + g) % 7) * 0.0007f + g * 0.0003f);
        }
        for (int d = 0; d < D; ++d) {
            const int magnitude = (d * 17 + row * 13) % 128;
            const int code = ((d + row) & 1) ? -magnitude : magnitude;
            cache_data.codes[row * D + d] = unwritten ? 0x7f : static_cast<std::uint8_t>(static_cast<std::int8_t>(code));
            cache_data.expanded[row * D + d] = f32_to_bf16(
                expand_ref(cache_data.codes[row * D + d], cache_data.scales[row * G + d / 64]));
        }
    }
    auto dc = to_device(cache_data.codes), ds = to_device(cache_data.scales), de = to_device(cache_data.expanded);
    GuardedDeviceBuffer out(query.size() * 2), reference(query.size() * 2);
    Tensor qt(dq.p, DType::BF16, {D, 4, tokens}), pt(dp.p, DType::I32, {tokens}),
        tt(dt.p, DType::I32, {static_cast<std::int32_t>(table.size())});
    Tensor ct(dc.p, DType::I8, {D, 64, Heads, Pages}), st(ds.p, DType::FP16, {G, 64, Heads, Pages});
    Tensor et(de.p, DType::BF16, {D, 64, Heads, Pages});
    Tensor ot(out.data(), DType::BF16, {D, 4, tokens}), rt(reference.data(), DType::BF16, {D, 4, tokens});
    ops::Gemma4CacheView cache{ops::Gemma4KvFormat::Int8Group64, &ct, &ct, &st, &st, &tt};
    ops::Gemma4CacheView bf16{ops::Gemma4KvFormat::BFloat16, &et, &et, nullptr, nullptr, &tt};
    WorkspaceArena workspace(std::max<std::size_t>(256, ops::gemma4_attention_workspace_capacity_bytes(D, 4, Heads, tokens)));
    ops::gemma4_attention(qt, cache, pt, 0, 1.0f, ot, workspace, nullptr);
    ops::gemma4_attention(qt, bf16, pt, 0, 1.0f, rt, workspace, nullptr);
    cuda_synchronize();
    const auto eager = from_device<std::uint16_t>(out.data(), query.size());
    int failures = verify_exact("int8 attention same represented BF16 values", eager,
                                from_device<std::uint16_t>(reference.data(), query.size()));
    for (auto bits : eager) {
        if (!std::isfinite(bf16_to_f32(bits))) {
            std::cerr << "int8 attention consumed poisoned unused tail\n"; ++failures; break;
        }
    }
    // Descriptor dispatch is fixed at capture; replay must reproduce the eager result.
    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture");
    ops::gemma4_attention(qt, cache, pt, 0, 1.0f, ot, workspace, stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
    cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "cudaGraphInstantiate");
    cuda_check(cudaGraphLaunch(executable, stream), "cudaGraphLaunch");
    cuda_synchronize(stream);
    failures += verify_exact("INT8 eager versus graph replay",
                              from_device<std::uint16_t>(out.data(), query.size()), eager);
    cuda_check(cudaGraphExecDestroy(executable), "cudaGraphExecDestroy");
    cuda_check(cudaGraphDestroy(graph), "cudaGraphDestroy");
    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    failures += out.verify_guards("INT8 attention") + reference.verify_guards("BF16 attention");
    return failures;
}

// Quantize real BF16 K/V rows with the host codec and compare INT8 attention to attention
// over the ORIGINAL BF16 cache: this bounds end-to-end error of the format itself.
int attention_vs_original_bf16_test(int tokens) {
    const std::vector<int> table{2, 0, 3, 1};
    std::vector<int> positions(tokens);
    for (int i = 0; i < tokens; ++i) { positions[i] = 65 + i; }
    std::vector<float> query(D * 4 * tokens), k(Rows * D), v(Rows * D);
    fill_uniform(query, 911, -0.08f, 0.08f);
    fill_uniform(k, 912, -1.0f, 1.0f);
    fill_uniform(v, 913, -1.0f, 1.0f);
    for (int row = 0; row < Rows; ++row) {
        for (int d = 0; d < D; ++d) {
            k[row * D + d] *= 0.25f + 0.35f * float(d / 64);   // different amplitude per group
            v[row * D + d] *= 0.2f + 0.1f * float(7 - d / 64);
        }
    }
    auto dq = to_device_bf16(query), dk = to_device_bf16(k), dv = to_device_bf16(v);
    const auto kb = from_device<std::uint16_t>(dk, k.size());
    const auto vb = from_device<std::uint16_t>(dv, v.size());
    std::vector<std::uint8_t> kc(Rows * D), vc(Rows * D);
    std::vector<std::uint16_t> kscale(Rows * G), vscale(Rows * G);
    for (int row = 0; row < Rows; ++row) {
        quantize(kb.data() + row * D, kc.data() + row * D, kscale.data() + row * G);
        quantize(vb.data() + row * D, vc.data() + row * D, vscale.data() + row * G);
    }
    auto dkc = to_device(kc), dvc = to_device(vc), dks = to_device(kscale), dvs = to_device(vscale);
    auto dp = to_device_i32(positions), dt = to_device_i32(table);
    GuardedDeviceBuffer out(query.size() * 2), reference(query.size() * 2);
    Tensor qt(dq.p, DType::BF16, {D, 4, tokens}), pt(dp.p, DType::I32, {tokens}), tt(dt.p, DType::I32, {Pages});
    Tensor kct(dkc.p, DType::I8, {D, 64, Heads, Pages}), vct(dvc.p, DType::I8, {D, 64, Heads, Pages});
    Tensor kst(dks.p, DType::FP16, {G, 64, Heads, Pages}), vst(dvs.p, DType::FP16, {G, 64, Heads, Pages});
    Tensor kbt(dk.p, DType::BF16, {D, 64, Heads, Pages}), vbt(dv.p, DType::BF16, {D, 64, Heads, Pages});
    Tensor ot(out.data(), DType::BF16, {D, 4, tokens}), rt(reference.data(), DType::BF16, {D, 4, tokens});
    ops::Gemma4CacheView cache{ops::Gemma4KvFormat::Int8Group64, &kct, &vct, &kst, &vst, &tt};
    ops::Gemma4CacheView bf16{ops::Gemma4KvFormat::BFloat16, &kbt, &vbt, nullptr, nullptr, &tt};
    WorkspaceArena workspace(std::max<std::size_t>(256, ops::gemma4_attention_workspace_capacity_bytes(D, 4, Heads, tokens)));
    ops::gemma4_attention(qt, cache, pt, 0, 1.0f, ot, workspace, nullptr);
    ops::gemma4_attention(qt, bf16, pt, 0, 1.0f, rt, workspace, nullptr);
    cuda_synchronize();
    return verify_pointwise("int8 attention vs original BF16 cache",
                            from_device_bf16(out.data(), query.size()),
                            from_device_bf16(reference.data(), query.size()),
                            PointwiseCriterion{0.012, 0.04}) +
           out.verify_guards("INT8 vs BF16") + reference.verify_guards("BF16 original");
}

int validation_test() {
    // Validation is host-only and never reads these deliberately opaque addresses.
    int storage = 0;
    Tensor k(&storage, DType::I8, {D, 64, Heads, Pages}), v = k;
    Tensor ks(&storage, DType::FP16, {G, 64, Heads, Pages}), vs = ks;
    Tensor table(&storage, DType::I32, {Pages});
    const ops::Gemma4CacheView valid{ops::Gemma4KvFormat::Int8Group64, &k, &v, &ks, &vs, &table};
    ops::detail::validate_gemma4_cache(valid, D, true, "test");
    int failures = 0;
    const auto rejects = [&](const ops::Gemma4CacheView& cache, int dim = D, bool global = true) {
        try { ops::detail::validate_gemma4_cache(cache, dim, global, "test"); }
        catch (const std::invalid_argument&) { return; }
        std::cerr << "invalid INT8 cache accepted\n"; ++failures;
    };
    auto bad = valid; bad.k_scales = nullptr; rejects(bad);
    bad = valid; bad.v_scales = nullptr; rejects(bad);
    bad = valid; bad.k_pages = nullptr; rejects(bad);
    bad = valid; bad.block_table = nullptr; rejects(bad);
    rejects(valid, 256); rejects(valid, D, false);
    // FP8 QK must never combine with INT8 codes.
    bad = valid; bad.qk_mode = ops::Gemma4QkMode::Fp8; rejects(bad);
    for (Tensor* plane : {&ks, &vs}) {
        const Tensor saved = *plane;
        plane->dtype = DType::BF16; rejects(valid); *plane = saved;
        plane->data = nullptr; rejects(valid); *plane = saved;
        plane->nb[1] += 2; rejects(valid); *plane = saved;
        for (int axis = 0; axis < 4; ++axis) {
            ++plane->ne[axis]; rejects(valid); *plane = saved;
        }
        plane->ne[0] = 2; plane->nb[1] = 2 * plane->nb[0]; // FP8 row-256 scale extent
        rejects(valid); *plane = saved;
    }
    for (Tensor* plane : {&k, &v}) {
        const Tensor saved = *plane;
        plane->dtype = DType::U8; rejects(valid); *plane = saved;   // FP8 code dtype
        plane->dtype = DType::BF16; rejects(valid); *plane = saved;
        plane->data = nullptr; rejects(valid); *plane = saved;
        ++plane->ne[3]; rejects(valid); *plane = saved;
    }
    Tensor bfk(&storage, DType::BF16, {D, 64, Heads, Pages});
    bad = valid; bad.format = ops::Gemma4KvFormat::BFloat16; bad.k_pages = &bfk; bad.v_pages = &bfk;
    rejects(bad); // stale INT8 scale planes are not admitted on BF16 descriptors.
    return failures;
}

int append_test() {
    // The external append API stays BF16-only; INT8 code pages must be rejected.
    std::vector<float> values(D * Heads, 0.5f);
    auto input = to_device_bf16(values);
    auto positions = to_device_i32({63});
    auto table = to_device_i32({0});
    GuardedDeviceBuffer kp(D * 64 * Heads * 2), vp(kp.bytes());
    Tensor x(input.p, DType::BF16, {D, Heads, 1}), p(positions.p, DType::I32, {1}), t(table.p, DType::I32, {1});
    Tensor k(kp.data(), DType::BF16, {D, 64, Heads, 1}), v(vp.data(), DType::BF16, {D, 64, Heads, 1});
    int failures = 0;
    ops::gemma4_kv_append(x, x, p, k, v, t, nullptr);
    cuda_synchronize();
    for (Tensor* plane : {&k, &v}) {
        plane->dtype = DType::I8;
        try { ops::gemma4_kv_append(x, x, p, k, v, t, nullptr); ++failures; }
        catch (const std::invalid_argument&) {}
        plane->dtype = DType::BF16;
    }
    return failures + kp.verify_guards("external append K") + vp.verify_guards("external append V");
}
} // namespace

int main() {
    int failures = validation_test();
    if (cuda_unavailable()) { std::cout << "SKIP: no usable CUDA device\n"; return failures ? 1 : 77; }
    failures += codec_test() + qkv_test() + append_test();
    // Decode (width < 64) has two SIMT kernels; every attention case runs under both. The
    // environment (NINFER_GEMMA4_DECODE_ATTN) only picks the starting mode.
    const bool saved_decode_new = runtime_switches().gemma4_decode_attn_new;
    for (bool decode_new : {false, true}) {
        RuntimeSwitchFlags flags;
        flags.gemma4_decode_attn_new = decode_new;
        apply_runtime_switch_flags(flags);
        // Widths 1..16: causal tails, unmapped pages inside a tile, poisoned unwritten rows.
        for (int tokens = 1; tokens <= 16; ++tokens) { failures += attention_test(tokens); }
        failures += attention_test(16, false);
        for (int tokens : {1, 5, 16}) {
            // Captured grid, device-side empty partitions, all-empty block table.
            failures += attention_test(tokens, false, true);
            failures += attention_test(tokens, true, true, true);
        }
        for (int tokens : {63, 64, 65, 127, 128, 129}) { failures += attention_test(tokens); }
        failures += attention_test(129, false);
        for (int tokens : {1, 63, 64, 129}) { failures += attention_vs_original_bf16_test(tokens); }
        for (int tokens : {64, 65, 128, 129}) { failures += attention_test(tokens, false, true); }
        failures += attention_test(65, true, true, true);
    }
    RuntimeSwitchFlags restore;
    restore.gemma4_decode_attn_new = saved_decode_new;
    apply_runtime_switch_flags(restore);
    std::cout << (failures ? "FAIL" : "OK") << " gemma4_int8_kv\n";
    return failures ? 1 : 0;
}
