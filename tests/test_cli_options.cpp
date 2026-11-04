#include "options.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

ninfer::cli::Options parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

bool rejects(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures      = 0;
    using Qk = ninfer::ops::Gemma4QkMode;
    failures += check(parse({"ninfer", "model.ninfer", "--prompt", "hello"}).gemma4_qk == Qk::BFloat16, "Gemma QK default changed");
    for (const auto& mode : {"bf16", "fp8"}) {
        const auto options = parse({"ninfer", "model.ninfer", "--prompt", "hello", "--gemma4-qk", mode});
        failures += check(options.gemma4_qk == ninfer::parse_gemma4_qk(mode), "QK parser lost mode");
    }
    for (const auto& mode : {"", "int8", "FP8"}) {
        bool rejected = false;
        try { (void)parse({"ninfer", "model.ninfer", "--prompt", "hello", "--gemma4-qk", mode}); }
        catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "invalid QK mode accepted");
    }
    bool missing_qk_rejected = false;
    try { (void)parse({"ninfer", "model.ninfer", "--prompt", "hello", "--gemma4-qk"}); }
    catch (const std::invalid_argument&) { missing_qk_rejected = true; }
    failures += check(missing_qk_rejected, "missing QK value accepted");
    for (bool gemma : {false, true}) {
        for (auto storage : {ninfer::KvCacheStorage::BFloat16, ninfer::KvCacheStorage::Fp8E4M3Row256,
                             ninfer::KvCacheStorage::Int8Group64}) {
            bool rejected = false;
            try { ninfer::validate_gemma4_qk(Qk::Fp8, storage, gemma); }
            catch (const std::invalid_argument&) { rejected = true; }
            failures += check(rejected == (!gemma || storage != ninfer::KvCacheStorage::Fp8E4M3Row256),
                              "FP8 QK runtime compatibility gate mismatch");
            ninfer::validate_gemma4_qk(Qk::BFloat16, storage, gemma);
        }
    }
    const ninfer::cli::Options int8 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "int8"});
    failures += check(int8.kv_cache == ninfer::KvCacheStorage::Int8Group64 &&
                          int8.gemma4_qk == Qk::BFloat16,
                      "--kv-dtype int8 did not select INT8-G64 KV with the default BF16 QK");
    bool int8_qk_rejected = false;
    try {
        const auto combined = parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                     "--kv-dtype", "int8", "--gemma4-qk", "fp8"});
        ninfer::validate_gemma4_qk(combined.gemma4_qk, combined.kv_cache, true);
    } catch (const std::invalid_argument& error) {
        int8_qk_rejected = std::string_view(error.what()).find("int8") != std::string_view::npos;
    }
    failures += check(int8_qk_rejected, "INT8 KV with FP8 QK was not rejected with a clear error");
    const auto choice = parse({"ninfer", "model.ninfer", "--prompt", "hello", "--choice", "",
                               "--choice", "a|b", "--choice", "你好"});
    failures += check(choice.choices == std::vector<std::string>{"", "a|b", "你好"},
                      "choice flags changed literal text or order");
    failures +=
        check(parse({"ninfer", "model.ninfer", "--prompt", "hello", "--regex", ""}).regex == "",
              "empty regex must be distinct from no regex");
    failures += check(rejects([] {
                          (void)parse({"ninfer", "model.ninfer", "--prompt", "hello", "--regex", "",
                                       "--choice", "yes"});
                      }),
                      "regex and choice were accepted together");
    failures += check(rejects([] {
                          (void)parse({"ninfer", "model.ninfer", "--prompt", "hello", "--regex",
                                       "a", "--json-object"});
                      }),
                      "regex and JSON were accepted together");
    failures += check(
        parse({"ninfer", "model.ninfer", "--prompt", "hello", "--grammar-file", "answer.gbnf"})
                .grammar_path == "answer.gbnf",
        "grammar file argument was lost");
    failures +=
        check(parse({"ninfer", "model.ninfer", "--prompt", "hello", "--json-object"}).json_object,
              "JSON object CLI option lost");
    failures += check(
        parse({"ninfer", "model.ninfer", "--prompt", "hello", "--json-schema-file", "answer.json"})
                .json_schema_path == "answer.json",
        "schema file CLI option lost");
    failures += check(rejects([] {
                          (void)parse({"ninfer", "model.ninfer", "--prompt", "hello",
                                       "--json-object", "--grammar-file", "a.gbnf"});
                      }),
                      "conflicting CLI constraints accepted");
    const ninfer::cli::Options configured =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "37"});
    failures += check(configured.thinking_budget == 37,
                      "--thinking-budget did not preserve its positive value");
    failures +=
        check(ninfer::cli::usage_text("ninfer-cli").find("--thinking-budget") != std::string::npos,
              "CLI help omits --thinking-budget");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "0"});
                      }),
                      "zero --thinking-budget was accepted");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "8", "--no-thinking"});
                      }),
                      "--thinking-budget was accepted with --no-thinking");
    const ninfer::cli::Options with_effort =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "8",
               "--reasoning-effort", "medium"});
    failures += check(with_effort.thinking_budget == 8 && with_effort.reasoning_effort,
                      "thinking budget did not coexist with reasoning effort");
    const ninfer::cli::Options dflash_vision =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--vision", "--spec", "dflash",
               "--draft-tokens", "7"});
    failures += check(dflash_vision.enable_vision &&
                          dflash_vision.speculative.backend == ninfer::SpeculativeBackend::DFlash &&
                          dflash_vision.speculative.draft_tokens == 7,
                      "CLI did not preserve the combined DFlash and Vision startup features");
    const ninfer::cli::Options mtp_default =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec", "mtp"});
    failures += check(mtp_default.speculative.adaptive_draft_tokens &&
                          mtp_default.speculative.draft_tokens == 0,
                      "CLI MTP did not default to the adaptive draft count");
    failures += check(rejects([&] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                       "mtp", "--draft-tokens", "8"});
                      }),
                      "CLI accepted an unsupported MTP draft count");
    for (const auto k : {1U, 2U, 7U, 15U}) {
        const auto dflash2 = parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                    "dflash2", "--draft-tokens", std::to_string(k)});
        failures += check(dflash2.speculative.backend == ninfer::SpeculativeBackend::DFlash2 &&
                              dflash2.speculative.draft_tokens == k,
                          "CLI did not preserve the DFlash2 draft count");
    }
    for (const auto k : {0U, 16U}) {
        failures +=
            check(rejects([&] {
                      (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                   "dflash2", "--draft-tokens", std::to_string(k)});
                  }),
                  "CLI accepted an unsupported DFlash2 draft count");
    }
    const ninfer::cli::Options nvfp4 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "nvfp4"});
    failures += check(nvfp4.kv_cache == ninfer::KvCacheStorage::Nvfp4Group16,
                      "--kv-dtype nvfp4 did not select group-16 NVFP4 KV");
    const ninfer::cli::Options k8v4 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "k8v4"});
    failures += check(k8v4.kv_cache == ninfer::KvCacheStorage::Fp8KeyNvfp4Value,
                      "--kv-dtype k8v4 did not select asymmetric K8V4 KV");
    const std::string help = ninfer::cli::usage_text("ninfer-cli");
    failures +=
        check(help.find("nvfp4") != std::string::npos && help.find("k8v4") != std::string::npos,
              "CLI help omits a production KV storage mode");
    const ninfer::cli::Options logging =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--log-level", "debug"});
    failures += check(logging.log_level == ninfer::product::LogLevel::Debug,
                      "CLI log level was not parsed");
    failures += check(help.find("--log-level") != std::string::npos,
                      "CLI help omits the log-level control");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--log-level", "verbose"});
                      }),
                      "CLI accepted an unknown log level");
    failures +=
        check(rejects([] {
                  (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--top-k", "21"});
              }),
              "CLI accepted top_k beyond the executable candidate domain");
    return failures == 0 ? 0 : 1;
}
