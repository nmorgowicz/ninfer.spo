#include "models/gemma4/program/kv_layout.h"
#include "models/gemma4/program/program.h"

#include <iostream>
#include <stdexcept>

namespace {
int failures = 0;
void check(bool ok, const char* label) {
    if (!ok) { std::cerr << "FAIL " << label << '\n'; ++failures; }
}
template<class F> void rejects(F&& operation, const char* label) {
    try { operation(); check(false, label); }
    catch (const std::invalid_argument&) {}
}
}

int main() {
    using namespace ninfer;
    namespace g4 = models::gemma4;
    const auto bf16 = g4::kv_layer_layout(KvCacheStorage::BFloat16, true, 512, 4);
    const auto fp8 = g4::kv_layer_layout(KvCacheStorage::Fp8E4M3Row256, true, 512, 4);
    const auto sliding = g4::kv_layer_layout(KvCacheStorage::Fp8E4M3Row256, false, 256, 16);
    check(bf16.code_dtype == DType::BF16 && bf16.scale_groups == 0 &&
              bf16.page_bytes() == 524288, "BF16 global page accounting");
    check(fp8.code_dtype == DType::U8 && fp8.scale_groups == 2 &&
              fp8.code_page_bytes == 131072 && fp8.scale_page_bytes == 1024 &&
              fp8.page_bytes() == 264192, "FP8 codes and both FP16 scales are counted");
    check(sliding.format == ops::Gemma4KvFormat::BFloat16 &&
              sliding.page_bytes() == 1048576 && sliding.scale_page_bytes == 0,
          "FP8 request leaves the sliding ring BF16");
    for (const auto unsupported : {KvCacheStorage::Int8Group64, KvCacheStorage::Nvfp4Group16,
                                   KvCacheStorage::Fp8KeyNvfp4Value,
                                   KvCacheStorage::BFloat16KeyValue}) {
        rejects([&] { (void)g4::gemma4_kv_format(unsupported); }, "unsupported Gemma KV format");
    }
    rejects([&] { (void)g4::kv_layer_layout(KvCacheStorage::Fp8E4M3Row256, true, 256, 4); },
            "FP8 global D256 rejection");

    g4::LoadedText model;
    model.vocab = 8;
    model.hidden = 256;
    g4::Gemma4LayerWeights global;
    global.global_layer = true; global.head_dim = 512; global.kv_heads = 4;
    g4::Gemma4LayerWeights local;
    local.head_dim = 256; local.kv_heads = 16; local.window = 1024;
    // Workspace accounting needs BF16 projection metadata, but no device storage or execution.
    for (auto* layer : {&global, &local}) {
        layer->q_heads = 32;
        for (auto* weight : {&layer->qkv, &layer->o_proj, &layer->gate_up, &layer->down}) {
            weight->qtype = QType::BF16;
        }
    }
    model.output_head.qtype = QType::BF16;
    model.layers.assign(10, global);
    model.layers.insert(model.layers.end(), 50, local);
    check(g4::Program::kv_bytes_per_token(model) == 81920, "qualified BF16 global bytes/token");
    check(g4::Program::kv_bytes_per_token(model, KvCacheStorage::Fp8E4M3Row256) == 41280,
          "qualified FP8 global bytes/token includes 320 scale bytes");
    // fixed_bytes is intentionally format independent and conservative even at tiny capacity.
    const auto fixed = g4::Program::fixed_bytes(model, 128, 0);
    const auto snapshots = g4::Program::fixed_bytes(model, 128, 1) - fixed;
    check(snapshots == (900ULL << 20), "one slot retains the unquantized 18-page ring snapshot");

    const auto automatic = KvCapacityPolicy::automatic();
    const auto rounded = g4::kv_page_capacity(KvCapacityPolicy::explicit_capacity(65),
                                             65, 262144, 128, 1);
    check(rounded.tokens == 128 && rounded.pages == 2 && rounded.extra_tokens == 0,
          "explicit 65 rounds up to two actual pages");
    rejects([&] { (void)g4::kv_page_capacity(KvCapacityPolicy::explicit_capacity(65),
                                           65, 262144, 65, 0); },
            "explicit 65 rejects fit 65 instead of silently shrinking");
    const auto fitted = g4::kv_page_capacity(automatic, 262144, 262144, 131137, 2);
    check(fitted.tokens == 131136 && fitted.extra_tokens == 0, "auto floors fit to whole pages");
    const auto spare = g4::kv_page_capacity(KvCapacityPolicy::explicit_capacity(65),
                                           262144, 262144, 321, 4);
    check(spare.tokens == 128 && spare.extra_tokens == 192,
          "spare pages start after actual rounded live pages");
    const auto capped = g4::kv_page_capacity(automatic, 262144, 262144, 1000000, 4);
    check(capped.tokens == 262144 && capped.extra_tokens == 65536,
          "native context and spare pool cap");
    g4::validate_native_context(262144, 262144);
    rejects([&] { g4::validate_native_context(262145, 262144); }, "native context plus one guard");
    rejects([&] { (void)g4::kv_page_capacity(automatic, 262145, 262144, 1000000, 0); },
            "oversized auto request cannot hide behind fitting less memory");
    rejects([&] { (void)g4::kv_page_capacity(automatic, 64, 262144, 63, 0); }, "tiny fit rejected");
    std::cout << (failures ? "FAIL" : "OK") << " Gemma4 capacity\n";
    return failures ? 1 : 0;
}
