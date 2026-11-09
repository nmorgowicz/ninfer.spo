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

void prefill_auto_selection_thresholds() {
    using namespace ninfer;
    namespace g4 = models::gemma4;
    constexpr std::size_t headroom = 4096;
    constexpr std::size_t per_token = 16;
    constexpr std::uint32_t context = 4096;
    const auto fixed = [](std::uint32_t chunk) { return std::size_t{chunk} * 32; };
    const std::uint32_t ladder[] = {2048, 1024, 512, 256, 128};
    for (int i = 0; i < 5; ++i) {
        const auto width = ladder[i];
        const auto threshold = fixed(width) + headroom + context * per_token;
        const auto selected =
            g4::prefill_budget(kAutoPrefillChunk, context, threshold, headroom, per_token, fixed);
        check(selected.chunk == width && selected.fixed_bytes == fixed(width) &&
                  selected.fit_tokens == context, "auto exact selection threshold");
        if (i < 4) {
            check(g4::prefill_budget(kAutoPrefillChunk, context, threshold - 1,
                                     headroom, per_token, fixed).chunk == ladder[i + 1],
                  "auto one byte below selection threshold");
        } else {
            rejects([&] {
                (void)g4::prefill_budget(kAutoPrefillChunk, context, threshold - 1,
                                         headroom, per_token, fixed);
            }, "auto cannot shrink context to fit minimum width");
        }
    }
    for (const auto free : {std::size_t{0}, headroom - 1}) {
        rejects([&] { (void)g4::prefill_budget(kAutoPrefillChunk, context, free,
                                               headroom, per_token, fixed); },
                "auto subtraction cannot underflow");
    }
    const auto explicit_width =
        g4::prefill_budget(384, context, 0, headroom, per_token, fixed);
    check(explicit_width.chunk == 384 && explicit_width.fixed_bytes == fixed(384) &&
              explicit_width.fit_tokens == 0, "explicit integer bypasses auto ladder");
    rejects([&] { (void)g4::prefill_budget(kAutoPrefillChunk, context, 1000000,
                                           headroom, 0, fixed); },
            "zero KV byte cost rejected");
    validate_prefill_chunk_architecture(kAutoPrefillChunk, true);
    validate_prefill_chunk_architecture(1024, false);
    bool clear_rejection = false;
    try { validate_prefill_chunk_architecture(kAutoPrefillChunk, false); }
    catch (const std::invalid_argument& error) {
        clear_rejection = std::string_view(error.what()).find("only for Gemma4") !=
                          std::string_view::npos;
    }
    check(clear_rejection, "non-Gemma auto rejected clearly");
}

void prefill_auto_capacity_policy() {
    using namespace ninfer;
    namespace g4 = models::gemma4;
    const auto policy = KvCapacityPolicy::automatic(12345);
    const auto capacity = g4::prefill_kv_capacity(policy, 65, 262144, 256, 1, true);
    check(capacity.tokens == 128 && capacity.extra_tokens == 128 &&
              policy.mode == KvCapacityMode::Automatic &&
              policy.automatic_headroom_bytes == 12345,
          "auto chunk preserves full non-page-aligned context, checkpoint pool and KV policy");
    rejects([&] { (void)g4::prefill_kv_capacity(policy, 4096, 262144, 4095, 0, true); },
            "auto KV policy cannot shrink requested context for auto chunk");
    check(g4::prefill_kv_capacity(policy, 4096, 262144, 4095, 0, false).tokens == 4032,
          "explicit chunk retains existing auto KV context behavior");
    rejects([&] {
        (void)g4::prefill_kv_capacity(KvCapacityPolicy::explicit_capacity(64),
                                     128, 262144, 256, 0, true);
    }, "auto chunk rejects explicit KV capacity smaller than caller context");
    check(g4::prefill_kv_capacity(KvCapacityPolicy::explicit_capacity(4096),
                                  4096, 262144, 8192, 1, true).extra_tokens == 4096,
          "auto chunk retains requested checkpoint slot budget");
}
}

int main() {
    using namespace ninfer;
    namespace g4 = models::gemma4;
    prefill_auto_selection_thresholds();
    prefill_auto_capacity_policy();
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
    const auto int8 = g4::kv_layer_layout(KvCacheStorage::Int8Group64, true, 512, 4);
    const auto int8_sliding = g4::kv_layer_layout(KvCacheStorage::Int8Group64, false, 256, 16);
    check(int8.format == ops::Gemma4KvFormat::Int8Group64 && int8.code_dtype == DType::I8 &&
              int8.scale_groups == 8 && int8.code_page_bytes == 131072 &&
              int8.scale_page_bytes == 4096 && int8.page_bytes() == 270336,
          "INT8 codes and all eight FP16 group scales are counted");
    check(int8_sliding.format == ops::Gemma4KvFormat::BFloat16 &&
              int8_sliding.page_bytes() == 1048576 && int8_sliding.scale_page_bytes == 0,
          "INT8 request leaves the sliding ring BF16");
    rejects([&] { (void)g4::kv_layer_layout(KvCacheStorage::Int8Group64, true, 256, 4); },
            "INT8 global D256 rejection");
    for (const auto unsupported : {KvCacheStorage::Nvfp4Group16,
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
    check(g4::Program::kv_bytes_per_token(model, KvCacheStorage::Int8Group64) == 42240,
          "qualified INT8 global bytes/token includes 1280 scale bytes");
    // fixed_bytes is intentionally format independent and conservative even at tiny capacity.
    const auto fixed = g4::Program::fixed_bytes(model, 128, 0);
    const auto snapshots = g4::Program::fixed_bytes(model, 128, 1) - fixed;
    check(snapshots == (900ULL << 20), "one slot retains the unquantized 18-page ring snapshot");
    for (const auto storage : {KvCacheStorage::BFloat16, KvCacheStorage::Fp8E4M3Row256,
                               KvCacheStorage::Int8Group64}) {
        const auto per_token = g4::Program::kv_bytes_per_token(model, storage);
        constexpr std::size_t headroom = 1ULL << 30;
        constexpr std::uint32_t context = 8192;
        const auto actual_fixed = [&](std::uint32_t chunk) {
            return g4::Program::fixed_bytes(model, chunk, 1, context / 64);
        };
        const auto free = actual_fixed(512) + headroom + context * per_token;
        const auto selected = g4::prefill_budget(kAutoPrefillChunk, context, free,
                                                 headroom, per_token, actual_fixed);
        check(selected.chunk == 512 && selected.fixed_bytes == actual_fixed(512) &&
                  selected.fit_tokens == context,
              "auto consumes actual Program fixed bytes and chosen KV precision");
        const auto preserved = g4::prefill_kv_capacity(KvCapacityPolicy::automatic(headroom),
                                                       context, 262144, selected.fit_tokens, 1, true);
        check(preserved.tokens == context && preserved.extra_tokens == 0,
              "real Program budget cannot consume reserved headroom or shrink context");
    }

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
