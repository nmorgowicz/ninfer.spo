#include "core/console_output.h"
#include "corpus.h"
#include "evaluation.h"
#include "kl.h"

#include "ninfer/engine.h"
#include "product/host_options.h"
#include "product/logging/logging.h"
#include "product/logging/pretty_format.h"
#include "product/logging/startup_log.h"

#include <nlohmann/json.hpp>
#include <spdlog/logger.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <memory>
#include <span>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using json  = nlohmann::json;
using ninfer::perplexity::CorpusSelection;
using ninfer::perplexity::ScoreAggregate;
using ninfer::perplexity::WindowPlan;

struct Options {
    bool help_requested = false;
    std::filesystem::path artifact;
    std::optional<std::filesystem::path> corpus;
    std::optional<std::filesystem::path> text;
    std::optional<std::filesystem::path> output;
    std::optional<std::filesystem::path> dump_topk;
    std::vector<ninfer::TokenId> prefix;
    std::uint32_t context               = 4096;
    std::uint32_t stride                = 2048;
    int device                          = 0;
    ninfer::KvCacheStorage kv           = ninfer::KvCacheStorage::Fp8E4M3Row256;
    ninfer::ops::Gemma4QkMode gemma4_qk = ninfer::ops::Gemma4QkMode::BFloat16;
    bool quick                          = false;
    ninfer::product::LogLevel log_level = ninfer::product::LogLevel::Info;
    std::optional<std::filesystem::path> kl_reference_write;
    std::optional<std::filesystem::path> kl_reference;
    std::uint64_t kl_seed        = 20261105;
    std::uint32_t kl_resamples   = 1000;
    std::uint32_t prefill_chunk  = 1024;
    std::uint32_t headroom_mib   = 0;
};

std::string usage_text() {
    return "usage: ninfer-spo-perplexity <model.ninfer> "
           "(--corpus <manifest.json> [--quick] | --text <utf8-file>)\n"
           "       [--context N] [--stride N] [--prefill-chunk N] [--headroom-mib N] [--device N]\n"
           "       [--gemma4-qk bf16|fp8] (default bf16)\n"
           "       [--kv-dtype bf16|int8|fp8|nvfp4|k8v4] [--output <directory>]\n"
           "       [--dump-topk <file>] [--prefix-ids <id,id,...>]\n"
           "       [--kl-reference-write <cache> | --kl-reference <cache>] [--kl-seed N]\n"
           "       [--kl-resamples N]  (exact paired full-vocabulary KL; see docs/perplexity.md)\n"
           "       [--log-level trace|debug|info|warning|error|critical|off]\n";
}

[[noreturn]] void usage_error(std::string_view message) {
    throw std::invalid_argument(std::string(message));
}

template <class Integer>
Integer parse_integer(std::string_view text, const char* label) {
    Integer value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        usage_error(std::string("invalid ") + label + ": " + std::string(text));
    }
    return value;
}

Options parse_options(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        return Options{.help_requested = true};
    }
    if (argc < 2 || std::string_view(argv[1]).starts_with("--")) {
        usage_error("artifact path is required");
    }
    Options out;
    out.artifact = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string_view option = argv[i];
        const auto value              = [&](const char* label) -> std::string_view {
            if (++i >= argc) { usage_error(std::string(label) + " requires a value"); }
            return argv[i];
        };
        if (option == "--corpus") {
            out.corpus = std::filesystem::path(value("--corpus"));
        } else if (option == "--text") {
            out.text = std::filesystem::path(value("--text"));
        } else if (option == "--quick") {
            out.quick = true;
        } else if (option == "--context") {
            out.context = parse_integer<std::uint32_t>(value("--context"), "context");
        } else if (option == "--stride") {
            out.stride = parse_integer<std::uint32_t>(value("--stride"), "stride");
        } else if (option == "--prefill-chunk") {
            out.prefill_chunk = parse_integer<std::uint32_t>(value("--prefill-chunk"), "prefill chunk");
            if (out.prefill_chunk < 16) { usage_error("--prefill-chunk must be at least 16"); }
        } else if (option == "--headroom-mib") {
            out.headroom_mib = parse_integer<std::uint32_t>(value("--headroom-mib"), "headroom MiB");
        } else if (option == "--device") {
            out.device = parse_integer<int>(value("--device"), "device");
        } else if (option == "--gemma4-qk") {
            out.gemma4_qk = ninfer::parse_gemma4_qk(value("--gemma4-qk"));
        } else if (option == "--kv-dtype") {
            const std::string_view dtype = value("--kv-dtype");
            if (dtype == "bf16") {
                out.kv = ninfer::KvCacheStorage::BFloat16;
            } else if (dtype == "int8") {
                out.kv = ninfer::KvCacheStorage::Int8Group64;
            } else if (dtype == "fp8") {
                out.kv = ninfer::KvCacheStorage::Fp8E4M3Row256;
            } else if (dtype == "nvfp4") {
                out.kv = ninfer::KvCacheStorage::Nvfp4Group16;
            } else if (dtype == "k8v4") {
                out.kv = ninfer::KvCacheStorage::Fp8KeyNvfp4Value;
            } else {
                usage_error("--kv-dtype must be bf16, int8, fp8, nvfp4, or k8v4");
            }
        } else if (option == "--output") {
            out.output = std::filesystem::path(value("--output"));
        } else if (option == "--dump-topk") {
            out.dump_topk = std::filesystem::path(value("--dump-topk"));
        } else if (option == "--prefix-ids") {
            std::string_view list = value("--prefix-ids");
            out.prefix.clear();
            while (!list.empty()) {
                const auto comma = list.find(',');
                const auto id = parse_integer<ninfer::TokenId>(list.substr(0, comma), "prefix id");
                if (id < 0) { usage_error("--prefix-ids takes token ids"); }
                out.prefix.push_back(id);
                list = comma == std::string_view::npos ? std::string_view{} : list.substr(comma + 1);
            }
            if (out.prefix.empty()) { usage_error("--prefix-ids requires at least one id"); }
        } else if (option == "--kl-reference-write") {
            out.kl_reference_write = std::filesystem::path(value("--kl-reference-write"));
        } else if (option == "--kl-reference") {
            out.kl_reference = std::filesystem::path(value("--kl-reference"));
        } else if (option == "--kl-seed") {
            out.kl_seed = parse_integer<std::uint64_t>(value("--kl-seed"), "kl seed");
        } else if (option == "--kl-resamples") {
            out.kl_resamples = parse_integer<std::uint32_t>(value("--kl-resamples"), "kl resamples");
        } else if (option == "--log-level") {
            out.log_level = ninfer::product::parse_log_level(value("--log-level"));
        } else {
            usage_error("unknown option: " + std::string(option));
        }
    }
    if (out.corpus.has_value() == out.text.has_value()) {
        usage_error("exactly one of --corpus and --text is required");
    }
    if (out.quick && !out.corpus) { usage_error("--quick requires --corpus"); }
    if (out.kl_reference_write && out.kl_reference) {
        usage_error("--kl-reference-write and --kl-reference are mutually exclusive");
    }
    if ((out.kl_reference_write || out.kl_reference) && out.dump_topk) {
        usage_error("--dump-topk cannot be combined with the KL modes");
    }
    if (out.kl_resamples == 0) { usage_error("--kl-resamples must be positive"); }
    if (out.context < 2 || out.stride == 0 || out.stride >= out.context) {
        usage_error("context/stride must satisfy context>=2 and 1<=stride<context");
    }
    return out;
}

std::string kv_name(ninfer::KvCacheStorage value) {
    switch (value) {
    case ninfer::KvCacheStorage::BFloat16:
        return "bf16";
    case ninfer::KvCacheStorage::BFloat16KeyValue:
        return "bf16-kv";
    case ninfer::KvCacheStorage::Int8Group64:
        return "int8-g64";
    case ninfer::KvCacheStorage::Fp8E4M3Row256:
        return "fp8-e4m3-r256";
    case ninfer::KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case ninfer::KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    }
    throw std::logic_error("unknown KV dtype");
}

std::string safe_component(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const unsigned char c : value) {
        out.push_back(std::isalnum(c) || c == '-' || c == '_' || c == '.' ? static_cast<char>(c)
                                                                          : '-');
    }
    return out.empty() ? "unknown" : out;
}

std::string timestamp() {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    std::ostringstream out;
    out << std::put_time(&utc, "%Y%m%d-%H%M%S");
    return out.str();
}

std::filesystem::path prepare_output_directory(const Options& options,
                                               const ninfer::LoadSummary& load,
                                               const CorpusSelection& corpus) {
    std::filesystem::path output = options.output.value_or(
        std::filesystem::path("profiles/perplexity") / safe_component(load.model_name) /
        safe_component(load.prefill_signature) / kv_name(options.kv) /
        safe_component(corpus.corpus_id) / safe_component(corpus.mode) / timestamp());
    if (std::filesystem::exists(output)) {
        if (!std::filesystem::is_directory(output) ||
            std::filesystem::directory_iterator(output) != std::filesystem::directory_iterator()) {
            throw std::runtime_error("output directory exists and is not empty: " +
                                     output.string());
        }
    } else if (!std::filesystem::create_directories(output)) {
        throw std::runtime_error("cannot create output directory: " + output.string());
    }
    return std::filesystem::absolute(output).lexically_normal();
}

double seconds_since(Clock::time_point begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
}

json aggregate_json(const ScoreAggregate& value) {
    return json{{"scored_tokens", value.scored_tokens},
                {"total_nll", value.total_nll},
                {"mean_nll", value.mean_nll()},
                {"perplexity", value.ppl()}};
}

struct EvaluationStream {
    ninfer::perplexity::CorpusStream source;
    std::vector<ninfer::TokenId> tokens;
    std::vector<WindowPlan> windows;
};

// --dump-topk file: "NTOPK001", u32 k, u32 0, then per scored position in corpus order
// (streams, then windows, then targets) k i32 ids and k f32 natural-log probabilities.
constexpr std::uint32_t kDumpTopK = 64;

std::ofstream open_topk_dump(const std::filesystem::path& path) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) { throw std::runtime_error("cannot create top-k dump: " + path.string()); }
    const std::uint32_t header[2] = {kDumpTopK, 0};
    out.write("NTOPK001", 8);
    out.write(reinterpret_cast<const char*>(header), sizeof(header));
    return out;
}

void write_topk(std::ofstream& out, const ninfer::TopTokenScores& top) {
    const std::size_t positions = top.ids.size() / top.k;
    for (std::size_t p = 0; p < positions; ++p) {
        out.write(reinterpret_cast<const char*>(top.ids.data() + p * top.k),
                  static_cast<std::streamsize>(top.k * sizeof(ninfer::TokenId)));
        out.write(reinterpret_cast<const char*>(top.logprobs.data() + p * top.k),
                  static_cast<std::streamsize>(top.k * sizeof(float)));
    }
    if (!out) { throw std::runtime_error("cannot write top-k dump"); }
}

namespace kl = ninfer::perplexity::kl;

// Exact paired full-vocabulary KL. Receives each score tile's BF16 logits from the engine, turns
// them into F32 log-probabilities and either appends them to the reference cache or compares them
// with the cached reference rows. The cache stores no logits and is bound to the run protocol.
class KlRun {
public:
    KlRun(const Options& options, const ninfer::LoadSummary& load, const CorpusSelection& corpus,
          const std::vector<EvaluationStream>& streams, const ninfer::EngineOptions& engine,
          std::uint64_t total_positions)
        : options_(options), writing_(options.kl_reference_write.has_value()),
          total_(total_positions) {
        kl::Fnv64 hash;
        json stream_ids = json::array();
        for (const auto& stream : streams) {
            stream_ids.push_back({{"id", stream.source.id},
                                  {"domain", stream.source.domain},
                                  {"tokens", stream.tokens.size()}});
            hash.add(stream.tokens.data(), stream.tokens.size() * sizeof(ninfer::TokenId));
            for (const WindowPlan& window : stream.windows) {
                hash.add_value(window.input_begin);
                hash.add_value(window.input_end);
                hash.add_value(window.target_begin);
                hash.add_value(window.target_end);
            }
        }
        if (!options.prefix.empty()) {
            hash.add(options.prefix.data(), options.prefix.size() * sizeof(ninfer::TokenId));
        }
        std::ostringstream hex;
        hex << std::hex << std::setw(16) << std::setfill('0') << hash.value();
        protocol_ = {{"schema", "ninfer-kl-protocol/1"},
                     {"corpus_id", corpus.corpus_id},
                     {"corpus_mode", corpus.mode},
                     {"streams", std::move(stream_ids)},
                     {"token_window_hash_fnv64", hex.str()},
                     {"context_tokens", options.context},
                     {"stride_tokens", options.stride},
                     {"prefill_chunk_tokens", options.prefill_chunk},
                     {"prefix_ids", options.prefix},
                     {"kv_dtype", kv_name(options.kv)},
                     {"gemma4_qk", ninfer::gemma4_qk_name(engine.gemma4_qk)},
                     {"positions", total_positions}};
        std::error_code size_error;
        const auto bytes = std::filesystem::file_size(options.artifact, size_error);
        reference_ = {{"artifact",
                       std::filesystem::absolute(options.artifact).lexically_normal().string()},
                      {"artifact_file_bytes", size_error ? json(nullptr) : json(bytes)},
                      {"name", load.model_name},
                      {"architecture", load.architecture},
                      {"prefill_signature", load.prefill_signature},
                      {"formats", load.weight_formats}};
        stats_.kl.assign(total_, 0.0F);
        stats_.ref_lp.assign(total_, 0.0F);
        stats_.cand_lp.assign(total_, 0.0F);
        stats_.top1_agree.assign(total_, 1);
        scores_.k               = 0;
        scores_.logits_observer = [this](std::span<const std::uint16_t> logits, std::int32_t vocab,
                                         std::int32_t columns) { tile(logits, vocab, columns); };
    }

    [[nodiscard]] ninfer::TopTokenScores& scores() noexcept { return scores_; }

    void begin_window(std::span<const ninfer::TokenId> targets, const std::string& stream,
                      const std::string& domain) {
        targets_       = targets;
        stream_        = stream;
        domain_        = domain;
        window_cursor_ = 0;
        window_begin_  = position_;
    }

    void end_window(std::span<const float> engine_logprobs) {
        if (window_cursor_ != targets_.size() || engine_logprobs.size() != targets_.size()) {
            throw std::runtime_error("KL observer saw " + std::to_string(window_cursor_) +
                                     " columns for a window of " + std::to_string(targets_.size()) +
                                     " targets");
        }
        for (std::size_t i = 0; i < targets_.size(); ++i) {
            const double diff = std::abs(static_cast<double>(engine_logprobs[i]) -
                                         static_cast<double>(stats_.cand_lp[window_begin_ + i]));
            max_target_diff_  = std::max(max_target_diff_, diff);
        }
        // The host log-softmax of the BF16 logits must agree with the engine's own target
        // log-probability; a larger gap means the observer columns are misaligned with targets.
        if (max_target_diff_ > 0.05) {
            throw std::runtime_error("KL observer target log-probability disagrees with the "
                                     "engine by " + std::to_string(max_target_diff_));
        }
        windows_.push_back(
            kl::WindowRecord{.stream = stream_, .domain = domain_, .begin = window_begin_,
                             .count = targets_.size()});
    }

    [[nodiscard]] json finish(const std::filesystem::path& output_directory) {
        if (position_ != total_) {
            throw std::runtime_error("KL run scored " + std::to_string(position_) + " of " +
                                     std::to_string(total_) + " positions");
        }
        json out{{"mode", writing_ ? "reference_write" : "compare"},
                 {"protocol", protocol_},
                 {"target_logprob_max_abs_diff", max_target_diff_},
                 {"dtype", "f32"}};
        if (writing_) {
            writer_->finish();
            out["cache"]     = std::filesystem::absolute(*options_.kl_reference_write).string();
            out["positions"] = writer_->positions();
            return out;
        }
        if (reader_->consumed() != reader_->positions()) {
            throw std::runtime_error("KL reference cache holds " +
                                     std::to_string(reader_->positions()) +
                                     " positions but this run consumed " +
                                     std::to_string(reader_->consumed()));
        }
        out["reference_cache"]    = std::filesystem::absolute(*options_.kl_reference).string();
        out["reference_artifact"] = reader_->reference();
        out["candidate_artifact"] = reference_;
        out["statistics"]         = kl::summarize(stats_, windows_, options_.kl_resamples,
                                                  options_.kl_seed);
        const std::filesystem::path sidecar = output_directory / "kl_positions.f32";
        std::ofstream file(sidecar, std::ios::binary | std::ios::trunc);
        for (std::size_t i = 0; i < total_; ++i) {
            const float record[4] = {stats_.kl[i], stats_.ref_lp[i], stats_.cand_lp[i],
                                     static_cast<float>(stats_.top1_agree[i])};
            file.write(reinterpret_cast<const char*>(record), sizeof(record));
        }
        if (!file) { throw std::runtime_error("cannot write " + sidecar.string()); }
        out["per_position_file"] = {{"path", sidecar.string()},
                                    {"record", "f32 x4: kl, ref_target_lp, cand_target_lp, top1_agree"}};
        return out;
    }

private:
    void open(std::int32_t vocab) {
        vocab_             = vocab;
        protocol_["vocab"] = vocab;
        if (writing_) {
            writer_ = std::make_unique<kl::CacheWriter>(*options_.kl_reference_write, vocab,
                                                        protocol_, reference_);
        } else {
            reader_ = std::make_unique<kl::CacheReader>(*options_.kl_reference, vocab, protocol_);
        }
    }

    void tile(std::span<const std::uint16_t> logits, std::int32_t vocab, std::int32_t columns) {
        if (vocab_ == 0) { open(vocab); }
        if (vocab != vocab_ || logits.size() != static_cast<std::size_t>(vocab) * columns) {
            throw std::runtime_error("KL observer tile has an inconsistent shape");
        }
        if (window_cursor_ + static_cast<std::size_t>(columns) > targets_.size() ||
            position_ + static_cast<std::uint64_t>(columns) > total_) {
            throw std::runtime_error("KL observer received more columns than scored positions");
        }
        const std::size_t cells = static_cast<std::size_t>(vocab) * columns;
        cand_.resize(cells);
        kl::parallel_columns(columns, [&](std::int32_t begin, std::int32_t end) {
            for (std::int32_t c = begin; c < end; ++c) {
                kl::log_softmax_bf16(logits.data() + static_cast<std::size_t>(c) * vocab, vocab,
                                     cand_.data() + static_cast<std::size_t>(c) * vocab);
            }
        });
        if (writing_) {
            writer_->append(cand_, columns);
        } else {
            ref_.resize(cells);
            reader_->read(columns, ref_);
        }
        std::atomic<bool> bad_target{false};
        kl::parallel_columns(columns, [&](std::int32_t begin, std::int32_t end) {
            for (std::int32_t c = begin; c < end; ++c) {
                const float* q = cand_.data() + static_cast<std::size_t>(c) * vocab;
                const auto t   = targets_[window_cursor_ + static_cast<std::size_t>(c)];
                if (t < 0 || t >= vocab) {
                    bad_target = true;
                    continue;
                }
                const std::size_t at = position_ + static_cast<std::size_t>(c);
                stats_.cand_lp[at]   = q[t];
                if (!writing_) {
                    const float* r      = ref_.data() + static_cast<std::size_t>(c) * vocab;
                    stats_.ref_lp[at]   = r[t];
                    stats_.kl[at]       = static_cast<float>(kl::kl_divergence(r, q, vocab));
                    stats_.top1_agree[at] = kl::argmax(r, vocab) == kl::argmax(q, vocab) ? 1 : 0;
                }
            }
        });
        if (bad_target) { throw std::runtime_error("KL observer target id is outside the vocab"); }
        position_ += static_cast<std::uint64_t>(columns);
        window_cursor_ += static_cast<std::size_t>(columns);
    }

    const Options& options_;
    bool writing_;
    std::uint64_t total_;
    json protocol_;
    json reference_;
    ninfer::TopTokenScores scores_;
    std::unique_ptr<kl::CacheWriter> writer_;
    std::unique_ptr<kl::CacheReader> reader_;
    std::int32_t vocab_ = 0;
    std::vector<float> cand_;
    std::vector<float> ref_;
    kl::PositionStats stats_;
    std::vector<kl::WindowRecord> windows_;
    std::span<const ninfer::TokenId> targets_;
    std::string stream_;
    std::string domain_;
    std::size_t window_cursor_ = 0;
    std::uint64_t window_begin_ = 0;
    std::uint64_t position_     = 0;
    double max_target_diff_     = 0.0;
};

int run(const Options& options, const std::shared_ptr<spdlog::logger>& logger,
        ninfer::product::StartupLogRenderer& startup_log,
        const std::shared_ptr<ninfer::product::TerminalProgress>& progress) {
    const Clock::time_point total_started = Clock::now();
    ninfer::EngineOptions engine_options;
    engine_options.artifact_path    = options.artifact;
    engine_options.purpose          = ninfer::EnginePurpose::CausalScoring;
    engine_options.device           = options.device;
    engine_options.max_context = options.context + static_cast<std::uint32_t>(options.prefix.size());
    engine_options.kv_cache         = options.kv;
    engine_options.gemma4_qk        = options.gemma4_qk;
    engine_options.prefill_chunk    = options.prefill_chunk;
    engine_options.scoring_headroom_bytes = static_cast<std::size_t>(options.headroom_mib) << 20U;
    engine_options.startup_observer = startup_log.observer();
    ninfer::Engine engine(std::move(engine_options));
    const ninfer::LoadSummary load = engine.load_summary();
    startup_log.engine_ready(load);
    logger->info("Gemma QK requested {} | resolved {} | {}",
                 ninfer::gemma4_qk_name(options.gemma4_qk),
                 ninfer::gemma4_qk_name(engine.options().gemma4_qk),
                 ninfer::gemma4_qk_scope(engine.options().gemma4_qk));

    const Clock::time_point preflight_started = Clock::now();
    logger->info("preparing corpus");
    CorpusSelection corpus = options.corpus
                                 ? ninfer::perplexity::load_corpus(*options.corpus, options.quick)
                                 : ninfer::perplexity::load_custom_text(*options.text);
    std::vector<EvaluationStream> streams;
    streams.reserve(corpus.streams.size());
    std::uint64_t total_scored_tokens = 0;
    std::uint64_t total_input_tokens  = 0;
    std::uint64_t total_windows       = 0;
    for (auto& source : corpus.streams) {
        std::vector<ninfer::TokenId> tokens = engine.tokenize_text(source.text);
        if (tokens.size() < 2) {
            throw std::runtime_error("stream tokenized to fewer than two tokens: " + source.id);
        }
        std::vector<WindowPlan> windows =
            ninfer::perplexity::plan_windows(tokens.size(), options.context, options.stride);
        total_input_tokens += static_cast<std::uint64_t>(tokens.size());
        total_scored_tokens += static_cast<std::uint64_t>(tokens.size() - 1);
        total_windows += static_cast<std::uint64_t>(windows.size());
        streams.push_back(EvaluationStream{.source  = std::move(source),
                                           .tokens  = std::move(tokens),
                                           .windows = std::move(windows)});
    }
    const double preflight_seconds = seconds_since(preflight_started);
    logger->info("corpus ready | {} streams | {} input tokens | {} scored tokens | {} windows | {}",
                 ninfer::product::format_pretty_count(streams.size()),
                 ninfer::product::format_pretty_count(total_input_tokens),
                 ninfer::product::format_pretty_count(total_scored_tokens),
                 ninfer::product::format_pretty_count(total_windows),
                 ninfer::product::format_pretty_duration(preflight_seconds));

    const std::filesystem::path output_directory = prepare_output_directory(options, load, corpus);
    std::optional<std::ofstream> topk_dump;
    if (options.dump_topk) { topk_dump.emplace(open_topk_dump(*options.dump_topk)); }
    ninfer::TopTokenScores top{.k = kDumpTopK};
    std::unique_ptr<KlRun> kl_run;
    if (options.kl_reference_write || options.kl_reference) {
        kl_run = std::make_unique<KlRun>(options, load, corpus, streams, engine.options(),
                                         total_scored_tokens);
    }
    const Clock::time_point scoring_started = Clock::now();
    logger->info("scoring | {} streams | {} tokens | {} windows",
                 ninfer::product::format_pretty_count(streams.size()),
                 ninfer::product::format_pretty_count(total_scored_tokens),
                 ninfer::product::format_pretty_count(total_windows));
    Clock::time_point next_progress = scoring_started + std::chrono::seconds(10);
    ScoreAggregate overall;
    std::map<std::string, ScoreAggregate> domains;
    json stream_reports             = json::array();
    std::uint64_t completed_windows = 0;

    for (std::size_t stream_index = 0; stream_index < streams.size(); ++stream_index) {
        EvaluationStream& stream = streams[stream_index];
        std::ostringstream stream_status;
        stream_status << "  scoring [" << stream_index + 1 << '/' << streams.size() << "] "
                      << ninfer::product::format_pretty_text(stream.source.id) << " | "
                      << ninfer::product::format_pretty_count(stream.tokens.size()) << " tokens | "
                      << ninfer::product::format_pretty_count(stream.windows.size()) << " windows";
        if (progress->enabled()) {
            progress->update(stream_status.str());
        } else {
            logger->debug("{}", stream_status.str());
        }
        const Clock::time_point stream_started = Clock::now();
        ScoreAggregate stream_score;
        json window_reports = json::array();
        for (std::size_t window_index = 0; window_index < stream.windows.size(); ++window_index) {
            const WindowPlan& window = stream.windows[window_index];
            std::vector<ninfer::TokenId> input = options.prefix;
            input.insert(input.end(),
                         stream.tokens.begin() + static_cast<std::ptrdiff_t>(window.input_begin),
                         stream.tokens.begin() + static_cast<std::ptrdiff_t>(window.input_end));
            const auto first_target =
                static_cast<std::uint32_t>(window.first_target + options.prefix.size());
            const Clock::time_point window_started = Clock::now();
            std::vector<float> logprobs;
            const std::size_t window_targets = window.target_end - window.target_begin;
            ninfer::TopTokenScores* score_top =
                topk_dump ? &top : (kl_run ? &kl_run->scores() : nullptr);
            if (kl_run) {
                kl_run->begin_window(std::span<const ninfer::TokenId>(stream.tokens)
                                         .subspan(window.target_begin, window_targets),
                                     stream.source.id, stream.source.domain);
            }
            try {
                logprobs = engine.score_tokens(std::move(input), first_target, score_top);
            } catch (const std::exception& error) {
                throw std::runtime_error("scoring " + stream.source.id + " window " +
                                         std::to_string(window_index) + " failed: " + error.what());
            }
            const std::size_t expected = window.target_end - window.target_begin;
            if (logprobs.size() != expected) {
                throw std::runtime_error("scoring returned an invalid target count for " +
                                         stream.source.id);
            }
            if (topk_dump) { write_topk(*topk_dump, top); }
            if (kl_run) { kl_run->end_window(logprobs); }
            ScoreAggregate window_score;
            window_score.add(logprobs);
            stream_score.add(window_score);
            overall.add(window_score);
            domains[stream.source.domain].add(window_score);
            ++completed_windows;
            json window_report            = aggregate_json(window_score);
            window_report["index"]        = window_index;
            window_report["input_begin"]  = window.input_begin;
            window_report["input_end"]    = window.input_end;
            window_report["target_begin"] = window.target_begin;
            window_report["target_end"]   = window.target_end;
            window_report["first_target"] = window.first_target;
            window_report["seconds"]      = seconds_since(window_started);
            window_reports.push_back(std::move(window_report));

            if (Clock::now() >= next_progress) {
                const double elapsed = seconds_since(scoring_started);
                const double rate    = static_cast<double>(overall.scored_tokens) / elapsed;
                const std::uint64_t remaining = total_scored_tokens - overall.scored_tokens;
                const double eta = rate > 0 ? static_cast<double>(remaining) / rate : 0.0;
                std::ostringstream line;
                line << "scoring | " << ninfer::product::format_pretty_count(overall.scored_tokens)
                     << '/' << ninfer::product::format_pretty_count(total_scored_tokens)
                     << " tokens | " << completed_windows << '/' << total_windows
                     << " windows | PPL " << std::fixed << std::setprecision(4) << overall.ppl()
                     << " | " << ninfer::product::format_pretty_rate(rate, "tok") << " | elapsed "
                     << ninfer::product::format_pretty_duration(elapsed) << " | ETA "
                     << ninfer::product::format_pretty_duration(eta);
                if (progress->enabled()) {
                    progress->update("  " + line.str());
                } else {
                    logger->info("{}", line.str());
                }
                next_progress = Clock::now() + std::chrono::seconds(10);
            }
        }
        const double stream_seconds = seconds_since(stream_started);
        progress->clear();
        logger->info("[{}/{}] {} | {} scored tokens | PPL {:.6g} | {}", stream_index + 1,
                     streams.size(), ninfer::product::format_pretty_text(stream.source.id),
                     ninfer::product::format_pretty_count(stream_score.scored_tokens),
                     stream_score.ppl(), ninfer::product::format_pretty_duration(stream_seconds));
        json stream_report               = aggregate_json(stream_score);
        stream_report["id"]              = stream.source.id;
        stream_report["domain"]          = stream.source.domain;
        stream_report["path"]            = stream.source.path.string();
        stream_report["input_tokens"]    = stream.tokens.size();
        stream_report["unscored_tokens"] = 1;
        stream_report["seconds"]         = stream_seconds;
        stream_report["windows"]         = std::move(window_reports);
        stream_reports.push_back(std::move(stream_report));
    }

    const double scoring_seconds = seconds_since(scoring_started);
    progress->clear();
    logger->info("scoring complete | {} tokens | {} windows | PPL {:.6g} | {} | {}",
                 ninfer::product::format_pretty_count(overall.scored_tokens), completed_windows,
                 overall.ppl(), ninfer::product::format_pretty_duration(scoring_seconds),
                 ninfer::product::format_pretty_rate(
                     static_cast<double>(overall.scored_tokens) / scoring_seconds, "tok"));
    json domain_reports = json::array();
    for (const auto& [domain, aggregate] : domains) {
        json item      = aggregate_json(aggregate);
        item["domain"] = domain;
        domain_reports.push_back(std::move(item));
    }

    json report{
        {"schema_version", 2},
        {"metric",
         {{"name", "fixed-window truncated-context causal perplexity"}, {"log_base", "natural"}}},
        {"artifact",
         {{"path", std::filesystem::absolute(options.artifact).lexically_normal().string()},
          {"architecture", load.architecture},
          {"name", load.model_name},
          {"prefill_signature", load.prefill_signature},
          {"formats", load.weight_formats}}},
        {"corpus",
         {{"id", corpus.corpus_id},
          {"mode", corpus.mode},
          {"source", corpus.source.string()},
          {"stream_count", streams.size()}}},
        {"execution",
         {{"purpose", "causal_scoring"},
          {"device", options.device},
          {"context_tokens", options.context},
          {"stride_tokens", options.stride},
          {"prefix_ids", options.prefix},
          {"prefill_chunk_tokens", options.prefill_chunk},
          {"score_tile_tokens", 1024},
          {"gemma4_qk_requested", ninfer::gemma4_qk_name(options.gemma4_qk)},
          {"gemma4_qk_resolved", ninfer::gemma4_qk_name(engine.options().gemma4_qk)},
          {"gemma4_qk_scope", ninfer::gemma4_qk_scope(engine.options().gemma4_qk)},
          {"kv_dtype", kv_name(options.kv)}}},
        {"timing",
         {{"load_seconds", load.load_seconds},
          {"read_and_tokenize_seconds", preflight_seconds},
          {"score_seconds", scoring_seconds},
          {"total_seconds", seconds_since(total_started)},
          {"scored_tokens_per_second",
           static_cast<double>(overall.scored_tokens) / scoring_seconds}}},
        {"streams", std::move(stream_reports)},
        {"domains", std::move(domain_reports)},
        {"overall", aggregate_json(overall)},
    };
    if (kl_run) { report["kl"] = kl_run->finish(output_directory); }

    const std::filesystem::path temporary = output_directory / "report.json.tmp";
    const std::filesystem::path final     = output_directory / "report.json";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) { throw std::runtime_error("cannot create report: " + temporary.string()); }
        output << std::setw(2) << report << '\n';
        output.flush();
        if (!output) { throw std::runtime_error("cannot write report: " + temporary.string()); }
    }
    std::filesystem::rename(temporary, final);

    std::cout << "Perplexity result\n"
              << "artifact: " << load.model_name << '\n'
              << "kv: " << kv_name(options.kv) << ", corpus: " << corpus.corpus_id << " / "
              << corpus.mode << ", context/stride: " << options.context << '/' << options.stride
              << "\n\n";
    std::cout << std::left << std::setw(24) << "domain" << std::right << std::setw(16) << "tokens"
              << std::setw(16) << "mean_nll" << std::setw(16) << "ppl" << '\n';
    for (const auto& [domain, aggregate] : domains) {
        std::cout << std::left << std::setw(24) << domain << std::right << std::setw(16)
                  << aggregate.scored_tokens << std::setw(16) << std::fixed << std::setprecision(6)
                  << aggregate.mean_nll() << std::setw(16) << aggregate.ppl() << '\n';
    }
    std::cout << std::left << std::setw(24) << "overall" << std::right << std::setw(16)
              << overall.scored_tokens << std::setw(16) << std::fixed << std::setprecision(6)
              << overall.mean_nll() << std::setw(16) << overall.ppl() << "\n\n"
              << "score rate: " << std::setprecision(1)
              << static_cast<double>(overall.scored_tokens) / scoring_seconds << " tok/s\n"
              << "report: " << final << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    try {
        options = parse_options(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "ninfer-spo-perplexity: " << error.what() << '\n';
        std::cerr << usage_text();
        return 1;
    }
    if (options.help_requested) {
        std::cout << usage_text();
        return 0;
    }
    const ninfer::ConsoleOutputCodePage console_code_page;
    ninfer::HostResidencyOptions host_residency;
    ninfer::product::apply_host_tuning(host_residency, options.device);
    ninfer::product::LoggingRuntime logging(
        {.logger_name  = "ninfer-spo-perplexity",
         .level        = options.log_level,
         .presentation = ninfer::product::LogPresentation::Tool});
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    ninfer::product::StartupLogRenderer startup_log(logging);
    try {
        return run(options, logger, startup_log, logging.terminal_progress());
    } catch (const std::exception& error) {
        logging.terminal_progress()->clear();
        logger->error("{}", ninfer::product::format_pretty_text(error.what()));
        return 1;
    }
}
