#include "models/gemma4/program/program.h"
#include "models/gemma4/program/kv_layout.h"
#include "models/gemma4/program/prefix_store.h"
#include "models/gemma4/execution/draft_forward.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ninfer;
namespace g4 = models::gemma4;
int failures = 0;
void require(bool ok, const char* label) {
    if (!ok) { std::cerr << "FAIL " << label << '\n'; ++failures; }
}
void cuda(cudaError_t status) {
    if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
}

// Small real BF16 projections, two target layers and a two-layer assistant reading both cache
// formats. The cache, not the weights, is FP8. No external model or tokenizer is needed.
struct Fixture {
    std::vector<DeviceBuffer> weights;
    g4::LoadedText model;
    void* norm = nullptr;
    std::uint32_t seed = 7;
    Weight matrix(int rows, int cols) {
        std::vector<float> values(static_cast<std::size_t>(rows) * cols);
        test::fill_uniform(values, seed++, -0.08f, 0.08f);
        std::vector<std::uint16_t> bits(values.size());
        std::transform(values.begin(), values.end(), bits.begin(), test::f32_to_bf16);
        weights.emplace_back(bits.size() * 2);
        weights.back().copy_from_host(bits.data(), bits.size() * 2);
        return g4::bf16_matrix(weights.back().p, rows, cols);
    }
    g4::Gemma4LayerWeights layer(bool global, bool draft) {
        g4::Gemma4LayerWeights out;
        out.hidden = 256; out.intermediate = 256;
        out.head_dim = global ? 512 : 256; out.q_heads = 1; out.kv_heads = 1;
        out.global_layer = global; out.window = global ? 0 : 64;
        out.rotary_dim = global ? 128 : 256; out.rope_theta = 10000.0f;
        out.input_norm = norm; out.q_norm = norm; out.k_norm = norm;
        out.post_attn_norm = norm; out.pre_ff_norm = norm; out.post_ff_norm = norm;
        out.qkv = matrix(out.head_dim * (draft ? 1 : global ? 2 : 3), 256);
        out.o_proj = matrix(256, out.head_dim);
        out.gate_up = matrix(512, 256); out.down = matrix(256, 256);
        return out;
    }
    Fixture() {
        std::vector<std::uint16_t> ones(512, test::f32_to_bf16(1.0f));
        weights.emplace_back(ones.size() * 2);
        weights.back().copy_from_host(ones.data(), ones.size() * 2);
        norm = weights.back().p;
        model.hidden = 256; model.vocab = 8; model.final_norm = norm;
        model.token_embedding.table = matrix(8, 256);
        model.output_head = matrix(8, 256);
        model.layers.push_back(layer(true, false));
        model.layers.push_back(layer(false, false));
        model.draft = std::make_unique<g4::Gemma4DraftWeights>();
        auto& draft = *model.draft;
        draft.hidden = 256; draft.backbone = 256; draft.vocab = 8; draft.final_norm = norm;
        draft.input_projection = matrix(256, 512);
        draft.output_projection = matrix(256, 256); draft.output_head = matrix(8, 256);
        draft.layers.push_back(layer(true, true)); draft.layers.push_back(layer(false, true));
        draft.kv_source = {0, 1};
    }
};

struct Cache {
    std::vector<DeviceBuffer> codes, scales;
    std::vector<Tensor> code_tensors, scale_tensors;
    DeviceBuffer table_store{3 * sizeof(std::int32_t)};
    Tensor table;
    std::vector<g4::Gemma4CacheLayer> views;
    Cache(const g4::LoadedText& model, ops::Gemma4QkMode qk_mode) {
        const std::int32_t ids[] = {0, 1, 2};
        table_store.copy_from_host(ids, sizeof(ids));
        table = Tensor(table_store.p, DType::I32, {3});
        for (const auto& layer : model.layers) {
            const auto layout = g4::kv_layer_layout(KvCacheStorage::Fp8E4M3Row256,
                                                     layer.global_layer, layer.head_dim, 1);
            for (int plane = 0; plane != 2; ++plane) {
                codes.emplace_back(layout.code_page_bytes * 3); codes.back().fill(0);
                code_tensors.emplace_back(codes.back().p, layout.code_dtype,
                    std::initializer_list<std::int32_t>{layer.head_dim, 64, 1, 3});
                scales.emplace_back(); scale_tensors.emplace_back();
                if (layout.scale_groups) {
                    scales.back() = DeviceBuffer(layout.scale_page_bytes * 3); scales.back().fill(0);
                    scale_tensors.back() = Tensor(scales.back().p, DType::FP16,
                        {layout.scale_groups, 64, 1, 3});
                }
            }
        }
        for (std::size_t i = 0; i < model.layers.size(); ++i) {
            const bool global = model.layers[i].global_layer;
            views.push_back({.format = global ? ops::Gemma4KvFormat::Fp8E4m3Row256
                                               : ops::Gemma4KvFormat::BFloat16,
                .qk_mode = global ? qk_mode : ops::Gemma4QkMode::BFloat16,
                .k_pages = &code_tensors[2*i], .v_pages = &code_tensors[2*i+1],
                .k_scales = global ? &scale_tensors[2*i] : nullptr,
                .v_scales = global ? &scale_tensors[2*i+1] : nullptr, .block_table = &table});
        }
    }
};

void target_and_draft(Fixture& fixture, cudaStream_t stream, ops::Gemma4QkMode qk_mode) {
    Cache cache(fixture.model, qk_mode);
    require(cache.views[0].qk_mode == qk_mode &&
                cache.views[1].qk_mode == ops::Gemma4QkMode::BFloat16,
            "QK selector applies only to global cache");
    DeviceArena workspace(64U << 20U);
    ops::Bf16GemmContext gemm(stream);
    std::vector<std::int32_t> ids(65), positions(65);
    for (int i = 0; i < 65; ++i) { ids[i] = i % 8; positions[i] = i; }
    DeviceBuffer id_store(ids.size()*4), pos_store(positions.size()*4), hidden_store(256*65*2);
    id_store.copy_from_host(ids.data(), ids.size()*4);
    pos_store.copy_from_host(positions.data(), positions.size()*4);
    Tensor tokens(id_store.p, DType::I32, {65}), pos(pos_store.p, DType::I32, {65});
    Tensor hidden(hidden_store.p, DType::BF16, {256,65});
    g4::gemma4_layers(tokens, hidden, fixture.model.token_embedding, fixture.model.layers,
                      cache.views, pos, gemm, workspace, stream);
    cuda(cudaStreamSynchronize(stream));
    for (int plane = 0; plane != 2; ++plane) {
        std::vector<std::uint16_t> scales(cache.scales[plane].bytes/2);
        cache.scales[plane].copy_to_host(scales.data(), cache.scales[plane].bytes);
        require(std::any_of(scales.begin(), scales.end(), [](auto v) { return v != 0; }),
                "target writes nonzero FP16 K and V scales");
    }
    const auto hidden_values = test::from_device_bf16(hidden.data, 256*65);
    require(std::all_of(hidden_values.begin(), hidden_values.end(),
                        [](float v) { return std::isfinite(v); }), "mixed target forward is finite");
    // Snapshot every global plane; the assistant must not write target KV, codes OR scales.
    std::vector<std::vector<std::uint8_t>> before(4);
    const DeviceBuffer* planes[] = {&cache.codes[0], &cache.codes[1], &cache.scales[0], &cache.scales[1]};
    for (int i = 0; i < 4; ++i) {
        before[i].resize(planes[i]->bytes);
        planes[i]->copy_to_host(before[i].data(), before[i].size());
    }
    Tensor token(static_cast<std::int32_t*>(id_store.p)+64, DType::I32, {1});
    Tensor position(static_cast<std::int32_t*>(pos_store.p)+64, DType::I32, {1});
    DeviceBuffer carry_store(256*2), next_store(4);
    cuda(cudaMemcpy(carry_store.p, static_cast<std::uint16_t*>(hidden_store.p)+64*256,
                    256*2, cudaMemcpyDeviceToDevice));
    Tensor carry(carry_store.p, DType::BF16, {256,1}), next(next_store.p, DType::I32, {1});
    workspace.reset();
    g4::gemma4_draft_step(token, position, carry, next, fixture.model.token_embedding,
                          *fixture.model.draft, cache.views, gemm, workspace, stream);
    cuda(cudaStreamSynchronize(stream));
    std::int32_t next_id = -1; next_store.copy_to_host(&next_id, 4);
    require(next_id >= 0 && next_id < 8, "MTP reads mixed cache and produces a token");
    for (int i = 0; i < 4; ++i) {
        std::vector<std::uint8_t> after(planes[i]->bytes);
        planes[i]->copy_to_host(after.data(), after.size());
        require(after == before[i], "MTP leaves all target code and scale planes unchanged");
    }
}

void four_plane_checkpoint(cudaStream_t stream) {
    const auto layout = g4::kv_layer_layout(KvCacheStorage::Fp8E4M3Row256, true, 512, 1);
    const std::size_t strides[] = {layout.code_page_bytes, layout.code_page_bytes,
                                   layout.scale_page_bytes, layout.scale_page_bytes};
    std::vector<DeviceBuffer> buffers;
    g4::PrefixStore::Config config;
    config.pool_pages = 3; config.stream = stream;
    config.host_bytes = layout.page_bytes();
    for (auto stride : strides) {
        buffers.emplace_back(3*stride); buffers.back().fill(0);
        config.global.push_back({static_cast<std::uint8_t*>(buffers.back().p), stride});
    }
    g4::PrefixStore store(std::move(config));
    const int source = store.allocate_page();
    const int copy = store.allocate_page();
    for (int i = 0; i < 4; ++i) {
        cuda(cudaMemsetAsync(static_cast<std::uint8_t*>(buffers[i].p)+source*strides[i],
                             0x21+i, strides[i], stream));
    }
    store.copy_page(source, copy);
    cuda(cudaStreamSynchronize(stream));
    for (int i = 0; i < 4; ++i) {
        std::vector<std::uint8_t> bytes(strides[i]);
        buffers[i].copy_to_host(bytes.data(), bytes.size(), copy*strides[i]);
        require(std::all_of(bytes.begin(), bytes.end(), [i](auto v) { return v == 0x21+i; }),
                "partial-tail COW copies both code and both scale planes");
    }
    store.release(copy);
    std::vector<g4::PositionKey> keys(63, 42);
    require(store.capture(keys, std::span<const int>(&source,1), 0, 62), "host checkpoint capture");
    store.release(source);
    cuda(cudaStreamSynchronize(stream));
    for (auto& buffer : buffers) { buffer.fill(0); }
    const auto match = store.best_match(keys, 63);
    require(match && match->on_host, "four-plane checkpoint is retained on host");
    if (!match) { return; }
    std::vector<int> restored;
    (void)store.restore(*match, restored);
    cuda(cudaStreamSynchronize(stream));
    for (int i = 0; i < 4; ++i) {
        std::vector<std::uint8_t> bytes(strides[i]);
        buffers[i].copy_to_host(bytes.data(), bytes.size(), restored[0]*strides[i]);
        require(std::all_of(bytes.begin(), bytes.end(), [i](auto v) { return v == 0x21+i; }),
                "host checkpoint restores both code and both scale planes");
    }
    for (int page : restored) { store.release(page); }
}

void program_summary(Fixture& fixture, cudaStream_t stream) {
    for (auto qk : {ops::Gemma4QkMode::BFloat16, ops::Gemma4QkMode::Fp8})
    for (auto format : {KvCacheStorage::BFloat16, KvCacheStorage::Fp8E4M3Row256}) {
        if (qk == ops::Gemma4QkMode::Fp8 && format != KvCacheStorage::Fp8E4M3Row256) continue;
        g4::Program program(fixture.model,128,16,2,stream,
                            {.history=false,.extra_tokens=64},format,qk);
        const auto memory = program.memory_summary();
        const auto global = g4::kv_layer_layout(format,true,512,1).page_bytes()*3;
        const auto sliding = g4::kv_layer_layout(format,false,256,1).page_bytes()*2;
        require(memory.gemma4_kv && memory.gemma4_kv->requested == format &&
                    memory.gemma4_kv->global == format &&
                    memory.gemma4_kv->qk_mode == qk &&
                    std::string(gemma4_qk_scope(qk)).find(qk == ops::Gemma4QkMode::Fp8 ? "tokens>=64" : "Phase1") != std::string::npos &&
                    memory.gemma4_kv->sliding == KvCacheStorage::BFloat16 &&
                    memory.gemma4_kv->global_payload_bytes == global &&
                    memory.gemma4_kv->sliding_payload_bytes == sliding &&
                    memory.kv_payload_bytes == global+sliding && memory.kv_capacity == 128,
                "Program summary counts actual pool, scales and BF16 ring with MTP loaded");
    }
}
}

int main() {
    if (ninfer::test::cuda_unavailable()) { std::cout << "SKIP: no usable CUDA device\n"; return 77; }
    cudaStream_t stream = nullptr;
    try {
        cuda(cudaStreamCreate(&stream));
        {
            Fixture fixture;
            target_and_draft(fixture,stream,ops::Gemma4QkMode::BFloat16);
            target_and_draft(fixture,stream,ops::Gemma4QkMode::Fp8);
            four_plane_checkpoint(stream);
            program_summary(fixture,stream);
        }
        cuda(cudaStreamDestroy(stream));
    } catch (const std::exception& error) {
        if (stream) { cudaStreamDestroy(stream); }
        std::cerr << "EXC " << error.what() << '\n'; return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " Gemma4 FP8 runtime\n";
    return failures ? 1 : 0;
}
