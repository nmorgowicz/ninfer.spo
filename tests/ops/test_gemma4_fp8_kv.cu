#include "ninfer/ops/gemma4_attention.h"
#include "ninfer/ops/gemma4_kv_append.h"
#include "ninfer/ops/gemma4_qkv_prep.h"
#include "ops/kernel/gemma4_fp8_kv.cuh"
#include "ops/op_tester.h"
#include "ops/wrapper/gemma4_cache_validation.h"

#include <limits>

using namespace ninfer;
using namespace ninfer::test;

namespace {
constexpr int D = 512;
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
float code_value(std::uint8_t code) {
    const int exp = (code >> 3) & 15;
    const int mantissa = code & 7;
    if (exp == 15 && mantissa == 7) { return std::numeric_limits<float>::quiet_NaN(); }
    const float value = exp == 0 ? std::ldexp(float(mantissa), -9) :
                                  std::ldexp(1.0f + float(mantissa) / 8.0f, exp - 7);
    return code & 128 ? -value : value;
}
std::uint8_t encode(float x) {
    const float magnitude = std::fabs(x);
    int best = 0;
    float error = std::numeric_limits<float>::infinity();
    for (int code = 0; code <= 126; ++code) {
        const float next = std::fabs(magnitude - code_value(static_cast<std::uint8_t>(code)));
        if (next < error || (next == error && !(code & 1))) { best = code; error = next; }
    }
    return static_cast<std::uint8_t>(best | (std::signbit(x) ? 128 : 0));
}
void quantize(const std::uint16_t* x, std::uint8_t* codes, std::uint16_t* scales) {
    for (int group = 0; group < 2; ++group) {
        float maximum = 0.0f;
        bool poison = false;
        for (int i = 0; i < 256; ++i) {
            const float v = bf16_to_f32(x[group * 256 + i]);
            poison |= !std::isfinite(v);
            maximum = std::fmax(maximum, std::fabs(v));
        }
        const float raw = maximum / 448.0f;
        scales[group] = half_bits(poison ? 1.0f : maximum == 0.0f ? 0.0f :
                                  std::fmin(65504.0f, std::fmax(0x1p-24f, raw)));
        const float inverse = maximum == 0.0f ? 0.0f : 1.0f / half_value(scales[group]);
        for (int i = 0; i < 256; ++i) {
            codes[group * 256 + i] = poison ? 0x7f : maximum == 0.0f ? 0 :
                encode(bf16_to_f32(x[group * 256 + i]) * inverse);
        }
    }
}

__global__ void codec_kernel(const __nv_bfloat162* x, std::uint8_t* codes, __half* scales) {
    __nv_bfloat162 values[8];
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    for (int s = 0; s < 8; ++s) { values[s] = x[row * 256 + lane + 32 * s]; }
    ops::detail::gemma4_fp8_store(values, codes, scales, row, lane);
}
__global__ void expand_kernel(const std::uint8_t* codes, const __half* scales,
                              __nv_bfloat16* out, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { out[i] = ops::detail::gemma4_fp8_expand(codes[i], scales[i / 256]); }
}

int codec_test() {
    constexpr int count = 10 * D;
    std::vector<float> values(count, 0.0f);
    for (int row = 1; row < 10; ++row) {
        for (int i = 0; i < D; ++i) {
            const float multiplier = i < 256 ? 1.0f : 17.0f;
            values[row * D + i] = multiplier * float((i * 19) % 101 - 50) / 37.0f;
        }
    }
    for (int i = D; i < 2 * D; ++i) { values[i] *= 0x1p-100f; }
    for (int i = 2 * D; i < 3 * D; ++i) { values[i] *= 0x1p30f; }
    values[3 * D + 17] = std::numeric_limits<float>::quiet_NaN();
    values[4 * D + 301] = std::numeric_limits<float>::infinity();
    values[5 * D + 31] = -std::numeric_limits<float>::infinity();
    for (int i = 6 * D; i < 7 * D; ++i) { values[i] = std::numeric_limits<float>::quiet_NaN(); }
    // Half-scale rounding and row-half boundaries use deliberately different amplitudes.
    values[7 * D + 255] = 0.252f;
    values[7 * D + 256] = -15.125f;
    values[8 * D + 0] = -0.0f;
    auto input = to_device_bf16(values);
    const auto input_bits = from_device<std::uint16_t>(input, count);
    GuardedDeviceBuffer codes(count), scales(10 * 2 * 2), expanded(count * 2);
    codec_kernel<<<10, 32>>>(static_cast<const __nv_bfloat162*>(input.p),
        static_cast<std::uint8_t*>(codes.data()), static_cast<__half*>(scales.data()));
    cuda_check_last_launch("codec_kernel");
    expand_kernel<<<(count + 127) / 128, 128>>>(static_cast<const std::uint8_t*>(codes.data()),
        static_cast<const __half*>(scales.data()), static_cast<__nv_bfloat16*>(expanded.data()), count);
    cuda_synchronize();
    std::vector<std::uint8_t> expected_codes(count);
    std::vector<std::uint16_t> expected_scales(20);
    for (int row = 0; row < 10; ++row) {
        quantize(input_bits.data() + row * D, expected_codes.data() + row * D,
                 expected_scales.data() + row * 2);
    }
    int failures = verify_exact("codec represented scales", from_device<std::uint16_t>(scales.data(), 20), expected_scales);
    failures += verify_exact("codec codes/nonfinite policy", from_device<std::uint8_t>(codes.data(), count), expected_codes);
    const auto got = from_device<std::uint16_t>(expanded.data(), count);
    for (int i = 0; i < count; ++i) {
        const float reference = code_value(expected_codes[i]) * half_value(expected_scales[i / 256]);
        if (std::isnan(reference) ? !std::isnan(bf16_to_f32(got[i])) : got[i] != f32_to_bf16(reference)) {
            std::cerr << "codec FP32-product/BF16 expansion mismatch at " << i << '\n';
            ++failures; break;
        }
    }
    failures += codes.verify_guards("codes") + scales.verify_guards("scales") + expanded.verify_guards("expanded");
    return failures;
}

int expansion_boundary_test() {
    // Find an actual double-rounding boundary: FP32 product -> BF16 must differ
    // from FP16 product -> BF16, so a regression to the shared FP16 helper fails.
    std::uint16_t chosen_scale = 0;
    std::uint8_t chosen_code = 0;
    for (int scale = 0x1001; scale < 0x3c00 && !chosen_scale; ++scale) {
        for (int code = 1; code <= 126; ++code) {
            const float product = code_value(static_cast<std::uint8_t>(code)) *
                                  half_value(static_cast<std::uint16_t>(scale));
            if (f32_to_bf16(product) != f32_to_bf16(half_value(half_bits(product)))) {
                chosen_scale = static_cast<std::uint16_t>(scale);
                chosen_code = static_cast<std::uint8_t>(code);
                break;
            }
        }
    }
    if (!chosen_scale) { std::cerr << "no expansion boundary found\n"; return 1; }
    auto codes = to_device(std::vector<std::uint8_t>(256, chosen_code));
    auto scales = to_device(std::vector<std::uint16_t>{chosen_scale});
    GuardedDeviceBuffer expanded(256 * 2);
    expand_kernel<<<2, 128>>>(static_cast<const std::uint8_t*>(codes.p),
        static_cast<const __half*>(scales.p), static_cast<__nv_bfloat16*>(expanded.data()), 256);
    cuda_synchronize();
    const auto expected = f32_to_bf16(code_value(chosen_code) * half_value(chosen_scale));
    return verify_exact("FP32 not FP16 dequant product",
        from_device<std::uint16_t>(expanded.data(), 256), std::vector<std::uint16_t>(256, expected)) +
        expanded.verify_guards("expansion boundary");
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
    GuardedDeviceBuffer k(Rows * D), v(Rows * D), ks(Rows * 4), vs(Rows * 4);
    GuardedDeviceBuffer kr(Rows * D * 2), vr(Rows * D * 2);
    std::vector<std::uint8_t> background(Rows * D, 0xa5);
    std::vector<std::uint16_t> scale_background(Rows * 2, 0x3555), bf_background(Rows * D, 0x3f80);
    k.copy_from_host(background.data(), k.bytes()); v.copy_from_host(background.data(), v.bytes());
    ks.copy_from_host(scale_background.data(), ks.bytes()); vs.copy_from_host(scale_background.data(), vs.bytes());
    kr.copy_from_host(bf_background.data(), kr.bytes()); vr.copy_from_host(bf_background.data(), vr.bytes());
    Tensor x(dx.p, DType::BF16, {(4 + Heads) * D, tokens}), w(dw.p, DType::BF16, {D});
    Tensor p(dp.p, DType::I32, {tokens}), t(dt.p, DType::I32, {Pages});
    Tensor qo(q.data(), DType::BF16, {D, 4, tokens}), qro(qr.data(), DType::BF16, {D, 4, tokens});
    Tensor kt(k.data(), DType::U8, {D, 64, Heads, Pages}), vt(v.data(), DType::U8, {D, 64, Heads, Pages});
    Tensor kst(ks.data(), DType::FP16, {2, 64, Heads, Pages}), vst(vs.data(), DType::FP16, {2, 64, Heads, Pages});
    Tensor krt(kr.data(), DType::BF16, {D, 64, Heads, Pages}), vrt(vr.data(), DType::BF16, {D, 64, Heads, Pages});
    ops::Gemma4CacheView cache{ops::Gemma4KvFormat::Fp8E4m3Row256, &kt, &vt, &kst, &vst, &t};
    ops::Gemma4CacheView reference{ops::Gemma4KvFormat::BFloat16, &krt, &vrt, nullptr, nullptr, &t};
    ops::gemma4_qkv_prep(x, w, w, p, 128, 1000000.0f, 1e-6f, true, qo, cache, nullptr);
    ops::gemma4_qkv_prep(x, w, w, p, 128, 1000000.0f, 1e-6f, true, qro, reference, nullptr);
    cuda_synchronize();
    int failures = verify_exact("qkv Q unchanged", from_device<std::uint16_t>(q.data(), q.bytes() / 2),
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
            quantize(kb.data() + row * D, ek.data() + row * D, eks.data() + row * 2);
            quantize(vb.data() + row * D, ev.data() + row * D, evs.data() + row * 2);
        }
    }
    failures += verify_exact("qkv K codes after norm/rope", from_device<std::uint8_t>(k.data(), ek.size()), ek);
    failures += verify_exact("qkv V codes from raw K", from_device<std::uint8_t>(v.data(), ev.size()), ev);
    failures += verify_exact("qkv K scales", from_device<std::uint16_t>(ks.data(), eks.size()), eks);
    failures += verify_exact("qkv V scales", from_device<std::uint16_t>(vs.data(), evs.size()), evs);
    for (auto* buffer : {&q, &k, &v, &ks, &vs}) { failures += buffer->verify_guards("qkv FP8"); }
    return failures;
}

int attention_test(int tokens, bool invalid_high_page = true) {
    // Sparse mapping exercises invalid pages, the page/row-half boundary and unused tail.
    const std::vector<int> table{2, -1, 0, invalid_high_page ? Pages : 1};
    std::vector<int> positions(tokens);
    for (int i = 0; i < tokens; ++i) { positions[i] = 65 + i; }
    std::vector<float> query(D * 4 * tokens);
    fill_uniform(query, 301, -0.05f, 0.05f);
    auto dq = to_device_bf16(query), dp = to_device_i32(positions), dt = to_device_i32(table);
    std::vector<std::uint8_t> codes(Rows * D);
    std::vector<std::uint16_t> scales(Rows * 2), expanded(Rows * D);
    for (int row = 0; row < Rows; ++row) {
        const int last = positions.back();
        const bool unwritten = table[last / 64] == row / (64 * Heads) &&
                               row % 64 > last % 64;
        scales[row * 2] = half_bits(0.00314159f + (row % 7) * 0.001f);
        scales[row * 2 + 1] = half_bits(0.0253f + (row % 5) * 0.01f);
        if (unwritten) { scales[row * 2] = scales[row * 2 + 1] = 0x7e00; }
        for (int d = 0; d < D; ++d) {
            codes[row * D + d] = static_cast<std::uint8_t>(((d * 17 + row * 13) % 120) | ((d + row) & 1 ? 128 : 0));
            if (unwritten) { codes[row * D + d] = 0x7f; }
            expanded[row * D + d] = f32_to_bf16(code_value(codes[row * D + d]) * half_value(scales[row * 2 + d / 256]));
        }
    }
    auto dc = to_device(codes), ds = to_device(scales), de = to_device(expanded);
    GuardedDeviceBuffer out(query.size() * 2), reference(query.size() * 2);
    Tensor qt(dq.p, DType::BF16, {D, 4, tokens}), pt(dp.p, DType::I32, {tokens}), tt(dt.p, DType::I32, {Pages});
    Tensor ct(dc.p, DType::U8, {D, 64, Heads, Pages}), st(ds.p, DType::FP16, {2, 64, Heads, Pages});
    Tensor et(de.p, DType::BF16, {D, 64, Heads, Pages});
    Tensor ot(out.data(), DType::BF16, {D, 4, tokens}), rt(reference.data(), DType::BF16, {D, 4, tokens});
    ops::Gemma4CacheView cache{ops::Gemma4KvFormat::Fp8E4m3Row256, &ct, &ct, &st, &st, &tt};
    ops::Gemma4CacheView bf16{ops::Gemma4KvFormat::BFloat16, &et, &et, nullptr, nullptr, &tt};
    WorkspaceArena workspace(std::max<std::size_t>(256, ops::gemma4_attention_workspace_capacity_bytes(D, 4, Heads, tokens)));
    ops::gemma4_attention(qt, cache, pt, 0, 1.0f, ot, workspace, nullptr);
    ops::gemma4_attention(qt, bf16, pt, 0, 1.0f, rt, workspace, nullptr);
    cuda_synchronize();
    int failures = verify_exact("attention same represented BF16 values", from_device<std::uint16_t>(out.data(), query.size()),
                                 from_device<std::uint16_t>(reference.data(), query.size()));
    for (auto bits : from_device<std::uint16_t>(out.data(), query.size())) {
        if (!std::isfinite(bf16_to_f32(bits))) {
            std::cerr << "attention consumed poisoned unused tail\n"; ++failures; break;
        }
    }
    // Capture the descriptor dispatch and replay without allocations or descriptor changes.
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
    failures += verify_exact("FP8 graph replay", from_device<std::uint16_t>(out.data(), query.size()),
                              from_device<std::uint16_t>(reference.data(), query.size()));
    cuda_check(cudaGraphExecDestroy(executable), "cudaGraphExecDestroy");
    cuda_check(cudaGraphDestroy(graph), "cudaGraphDestroy");
    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    failures += out.verify_guards("FP8 attention") + reference.verify_guards("BF16 attention");
    return failures;
}

int validation_test() {
    // Validation is host-only and never reads these deliberately opaque addresses.
    int storage = 0;
    Tensor k(&storage, DType::U8, {D, 64, Heads, Pages}), v = k;
    Tensor ks(&storage, DType::FP16, {2, 64, Heads, Pages}), vs = ks;
    Tensor table(&storage, DType::I32, {Pages});
    const ops::Gemma4CacheView valid{ops::Gemma4KvFormat::Fp8E4m3Row256, &k, &v, &ks, &vs, &table};
    ops::detail::validate_gemma4_cache(valid, D, true, "test");
    int failures = 0;
    const auto rejects = [&](const ops::Gemma4CacheView& cache, int dim = D, bool global = true) {
        try { ops::detail::validate_gemma4_cache(cache, dim, global, "test"); }
        catch (const std::invalid_argument&) { return; }
        std::cerr << "invalid cache accepted\n"; ++failures;
    };
    auto bad = valid; bad.k_scales = nullptr; rejects(bad);
    bad = valid; bad.v_scales = nullptr; rejects(bad);
    bad = valid; bad.k_pages = nullptr; rejects(bad);
    bad = valid; bad.v_pages = nullptr; rejects(bad);
    bad = valid; bad.block_table = nullptr; rejects(bad);
    bad = valid; bad.format = static_cast<ops::Gemma4KvFormat>(99); rejects(bad);
    rejects(valid, 256); rejects(valid, D, false);
    for (Tensor* plane : {&ks, &vs}) {
        const Tensor saved = *plane;
        plane->dtype = DType::BF16; rejects(valid); *plane = saved;
        plane->data = nullptr; rejects(valid); *plane = saved;
        plane->nb[1] += 2; rejects(valid); *plane = saved;
        for (int axis = 0; axis < 4; ++axis) {
            ++plane->ne[axis]; rejects(valid); *plane = saved;
        }
    }
    for (Tensor* plane : {&k, &v}) {
        const Tensor saved = *plane;
        plane->dtype = DType::BF16; rejects(valid); *plane = saved;
        plane->data = nullptr; rejects(valid); *plane = saved;
        plane->nb[1] += 1; rejects(valid); *plane = saved;
        ++plane->ne[3]; rejects(valid); *plane = saved;
    }
    Tensor bfk(&storage, DType::BF16, {D, 64, Heads, Pages});
    bad = valid; bad.format = ops::Gemma4KvFormat::BFloat16; bad.k_pages = &bfk; bad.v_pages = &bfk;
    rejects(bad); // stale FP8 scale planes are not admitted on BF16 descriptors.
    bad.k_scales = bad.v_scales = nullptr;
    ops::detail::validate_gemma4_cache(bad, D, true, "test");
    return failures;
}

int append_test() {
    std::vector<float> values(D * Heads, 0.5f);
    auto input = to_device_bf16(values);
    auto positions = to_device_i32({63});
    auto table = to_device_i32({0});
    GuardedDeviceBuffer kp(D * 64 * Heads * 2), vp(kp.bytes());
    Tensor x(input.p, DType::BF16, {D, Heads, 1}), p(positions.p, DType::I32, {1}), t(table.p, DType::I32, {1});
    Tensor k(kp.data(), DType::BF16, {D, 64, Heads, 1}), v(vp.data(), DType::BF16, {D, 64, Heads, 1});
    ops::gemma4_kv_append(x, x, p, k, v, t, nullptr);
    cuda_synchronize();
    int failures = 0;
    const auto got = from_device<std::uint16_t>(kp.data(), kp.bytes() / 2);
    for (int h = 0; h < Heads; ++h) {
        for (int d = 0; d < D; ++d) {
            if (got[(h * 64 + 63) * D + d] != f32_to_bf16(0.5f)) { ++failures; break; }
        }
    }
    for (Tensor* plane : {&k, &v}) {
        plane->dtype = DType::U8;
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
    failures += codec_test() + expansion_boundary_test() + qkv_test() + append_test();
    for (int tokens : {1, 63, 64, 65, 129}) { failures += attention_test(tokens); }
    failures += attention_test(129, false);
    std::cout << (failures ? "FAIL" : "OK") << " gemma4_fp8_kv\n";
    return failures ? 1 : 0;
}
