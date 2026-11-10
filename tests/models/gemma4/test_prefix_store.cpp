// PrefixStore on tiny device buffers: a demoted checkpoint restores its global pages and ring
// snapshot byte for byte, a capture that makes room for the displaced state never drops the
// checkpoint about to be restored, and touch() keeps a checkpoint the prompts extend from being
// the first evicted.

#include "core/runtime_options.h"
#include "models/gemma4/program/prefix_cache.h"
#include "models/gemma4/program/prefix_store.h"
#include "models/gemma4/program/sliding_ring.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::models::gemma4::kKvPageTokens;
using ninfer::models::gemma4::PositionKey;
using ninfer::models::gemma4::PrefixStore;

constexpr int kPoolPages        = 32;
constexpr int kRingPages        = 8;
constexpr int kWindow           = 64;
constexpr std::size_t kPage     = 256;
constexpr std::uint32_t kTokens = 300;

int failures = 0;

void check(bool ok, const std::string& label) {
    std::cout << (ok ? "ok   " : "FAIL ") << label << '\n';
    if (!ok) { ++failures; }
}

void cuda(cudaError_t status) {
    if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
}

struct Buffers {
    std::uint8_t* global = nullptr;
    std::uint8_t* ring   = nullptr;
    cudaStream_t stream  = nullptr;
    Buffers() {
        cuda(cudaMalloc(&global, kPoolPages * kPage));
        cuda(cudaMalloc(&ring, kRingPages * kPage));
        cuda(cudaStreamCreate(&stream));
    }
    ~Buffers() {
        cudaFree(global);
        cudaFree(ring);
        cudaStreamDestroy(stream);
    }
    Buffers(const Buffers&)            = delete;
    Buffers& operator=(const Buffers&) = delete;
};

// Device slot of one snapshot plus five global pages, so the host arena holds two checkpoints.
std::size_t point_bytes() {
    const int snapshot = PrefixStore::snapshot_pages(kRingPages, kWindow);
    const auto pages   = (kTokens + kKvPageTokens - 1) / kKvPageTokens;
    return static_cast<std::size_t>(snapshot) * kPage + pages * kPage;
}

PrefixStore make_store(const Buffers& buffers) {
    PrefixStore::Config config;
    config.global       = {{buffers.global, kPage}};
    config.ring         = {{buffers.ring, kPage}};
    config.pool_pages   = kPoolPages;
    config.ring_pages   = kRingPages;
    config.window       = kWindow;
    config.device_slots = 1;
    config.host_bytes   = 2 * point_bytes();
    config.stream       = buffers.stream;
    return PrefixStore(std::move(config));
}

PrefixStore::Config host_config(const Buffers& buffers, std::size_t bytes) {
    PrefixStore::Config config;
    config.global     = {{buffers.global, kPage}};
    config.ring       = {{buffers.ring, kPage}};
    config.pool_pages = kPoolPages;
    config.ring_pages = kRingPages;
    config.window     = kWindow;
    config.host_bytes = bytes;
    config.stream     = buffers.stream;
    return config;
}

std::size_t ring_bytes() {
    return static_cast<std::size_t>(PrefixStore::snapshot_pages(kRingPages, kWindow)) * kPage;
}

std::vector<int> tagged_pages(PrefixStore& store, const Buffers& buffers, int count,
                              std::uint8_t tag) {
    std::vector<int> pages;
    for (int page = 0; page < count; ++page) {
        pages.push_back(store.allocate_page());
        cuda(cudaMemsetAsync(buffers.global + pages.back() * kPage, tag + page, kPage,
                             buffers.stream));
    }
    return pages;
}

bool tagged_page(const Buffers& buffers, int page, std::uint8_t tag,
                 std::size_t first = 0, std::size_t count = kPage) {
    std::vector<std::uint8_t> bytes(count);
    cuda(cudaStreamSynchronize(buffers.stream));
    cuda(cudaMemcpy(bytes.data(), buffers.global + page * kPage + first, count,
                    cudaMemcpyDeviceToHost));
    return std::all_of(bytes.begin(), bytes.end(), [tag](auto value) { return value == tag; });
}

bool tagged_ring(const Buffers& buffers, std::uint8_t tag, int first, int count) {
    std::vector<std::uint8_t> bytes(kPage);
    cuda(cudaStreamSynchronize(buffers.stream));
    for (int page = first; page < first + count; ++page) {
        cuda(cudaMemcpy(bytes.data(), buffers.ring + (page % kRingPages) * kPage, kPage,
                        cudaMemcpyDeviceToHost));
        if (!std::all_of(bytes.begin(), bytes.end(), [tag](auto value) { return value == tag; })) {
            return false;
        }
    }
    return true;
}

std::vector<PositionKey> keys_from(PositionKey first, std::uint32_t count) {
    std::vector<PositionKey> keys(count);
    std::iota(keys.begin(), keys.end(), first);
    return keys;
}

// Writes a sequence of kTokens positions tagged `tag` into fresh pool pages and the ring,
// captures it, and gives the pages back as the live sequence would on the next request.
void capture_sequence(PrefixStore& store, const Buffers& buffers, PositionKey first,
                      std::uint8_t tag, std::optional<std::uint64_t> keep = std::nullopt) {
    std::vector<int> live;
    for (std::uint32_t page = 0; page * kKvPageTokens < kTokens; ++page) {
        live.push_back(store.allocate_page());
        cuda(cudaMemsetAsync(buffers.global + live.back() * kPage, tag + page, kPage, buffers.stream));
    }
    for (int page = 0; page < kRingPages; ++page) {
        cuda(cudaMemsetAsync(buffers.ring + page * kPage, tag + 0x40 + page, kPage, buffers.stream));
    }
    store.capture(keys_from(first, kTokens), live, 0, kTokens - 1, keep);
    for (const int page : live) { store.release(page); }
}

std::vector<PositionKey> prompt_after(PositionKey first) {
    auto prompt = keys_from(first, kTokens);
    prompt.push_back(999999);
    return prompt;
}

// A, B, C captured in turn with one device slot: A and B are on the host (full), C on the device.
void fill(PrefixStore& store, const Buffers& buffers) {
    capture_sequence(store, buffers, 1000, 0x10);
    capture_sequence(store, buffers, 2000, 0x20);
    capture_sequence(store, buffers, 3000, 0x30);
}

void restores_kept_checkpoint() {
    Buffers buffers;
    PrefixStore store = make_store(buffers);
    fill(store, buffers);
    const auto match = store.best_match(prompt_after(1000), kTokens);
    check(match && match->on_host && match->reuse == kTokens, "A is matched on the host");
    if (!match) { return; }
    capture_sequence(store, buffers, 4000, 0x40, match->id);
    check(store.stats().drops == 1, "making room for D drops one host checkpoint");
    check(store.best_match(prompt_after(2000), kTokens) == std::nullopt, "B, not A, was dropped");
    std::vector<int> live;
    const std::int64_t valid_from = store.restore(*match, live);
    cuda(cudaStreamSynchronize(buffers.stream));
    bool same = valid_from == 0 && live.size() == 5;
    std::vector<std::uint8_t> host(kPage);
    for (std::size_t page = 0; same && page < live.size(); ++page) {
        cuda(cudaMemcpy(host.data(), buffers.global + live[page] * kPage, kPage, cudaMemcpyDeviceToHost));
        const auto tag = static_cast<std::uint8_t>(0x10 + page);
        same           = host[0] == tag && host[kPage - 1] == tag;
    }
    for (int page = 0; same && page < 5; ++page) {
        cuda(cudaMemcpy(host.data(), buffers.ring + page * kPage, kPage, cudaMemcpyDeviceToHost));
        same = host[0] == static_cast<std::uint8_t>(0x50 + page);
    }
    check(same, "A restores its global pages and ring snapshot");
    for (const int page : live) { store.release(page); }
}

void drops_without_keep() {
    Buffers buffers;
    PrefixStore store = make_store(buffers);
    fill(store, buffers);
    capture_sequence(store, buffers, 4000, 0x40);
    check(store.best_match(prompt_after(1000), kTokens) == std::nullopt,
          "without keep, the oldest host checkpoint A is dropped");
}

void touch_keeps_used_checkpoint() {
    Buffers buffers;
    PrefixStore store = make_store(buffers);
    fill(store, buffers);
    store.touch(prompt_after(1000));
    capture_sequence(store, buffers, 4000, 0x40);
    check(store.best_match(prompt_after(1000), kTokens).has_value() &&
              !store.best_match(prompt_after(2000), kTokens).has_value(),
          "a touched checkpoint outlives an untouched older one");
}

void host_delta_branches(bool delta) {
    Buffers buffers;
    PrefixStore store(host_config(buffers, 4 * point_bytes()));
    auto live = tagged_pages(store, buffers, 5, 0x10);
    auto a = keys_from(1000, kTokens);
    cuda(cudaMemsetAsync(buffers.ring, 0x50, kRingPages * kPage, buffers.stream));
    check(store.capture(a, live, 0, 299), "capture partial-page branch A");

    // Appending in the partial fifth page changes its identity, not the four complete pages.
    store.will_write(live[4]);
    cuda(cudaMemsetAsync(buffers.global + live[4] * kPage + 172, 0x99, 44, buffers.stream));
    cuda(cudaMemsetAsync(buffers.ring, 0x60, kRingPages * kPage, buffers.stream));
    auto b = keys_from(1000, 310);
    b[299] = 7777;
    check(store.capture(b, live, 0, 309), "capture extending branch B");
    const auto stats = store.stats();
    check(stats.host_global_copied_bytes == (delta ? 6 : 10) * kPage &&
              stats.host_global_reused_bytes == (delta ? 4 : 0) * kPage,
          "delta copies only the changed partial page; control copies all pages");
    check(stats.host_bytes_used == 2 * ring_bytes() + (delta ? 6 : 10) * kPage,
          "shared payload is charged once in the bounded arena");
    for (const int page : live) { store.release(page); }
    live.clear();

    const auto ma = store.best_match(a, kTokens);
    const auto mb = store.best_match(b, 310);
    check(ma && mb && ma->id != mb->id, "both branch frontiers remain independently matchable");
    if (!ma || !mb) { return; }
    check(store.restore(*ma, live) == 0, "A retains its original ring coverage");
    bool same = live.size() == 5 && tagged_ring(buffers, 0x50, 0, 5);
    for (std::size_t page = 0; same && page < live.size(); ++page) {
        same = tagged_page(buffers, live[page], static_cast<std::uint8_t>(0x10 + page));
    }
    check(same, "A restores every byte including its old partial-page suffix and own ring");
    for (const int page : live) { store.release(page); }
    live.clear();
    check(store.restore(*mb, live) == 0, "B restores after recycled pool pages were allocated");
    check(tagged_page(buffers, live[4], 0x14, 0, 172) &&
              tagged_page(buffers, live[4], 0x99, 172, 44) &&
              tagged_page(buffers, live[4], 0x14, 216, 40) &&
              tagged_ring(buffers, 0x60, 0, 5),
          "B restores its changed bytes and independent ring without corrupting A");

    // An explicit earlier boundary of a restored state can share the restored identities.
    const auto before = store.stats();
    b.resize(256);
    check(store.capture(b, live, 0, 309), "capture explicit boundary from restored B");
    check(store.stats().host_global_copied_bytes ==
                  before.host_global_copied_bytes + (delta ? 0 : 4 * kPage) &&
              store.stats().host_global_reused_bytes ==
                  before.host_global_reused_bytes + (delta ? 4 * kPage : 0),
          "host restore propagates identities to the new physical pages");
    for (const int page : live) { store.release(page); }
}

void delta_budget_and_keep() {
    Buffers buffers;
    const std::size_t budget = 2 * ring_bytes() + 6 * kPage;
    PrefixStore store(host_config(buffers, budget));
    auto live = tagged_pages(store, buffers, 5, 0x20);
    auto a = keys_from(2000, 300);
    cuda(cudaMemsetAsync(buffers.ring, 0x70, kRingPages * kPage, buffers.stream));
    check(store.capture(a, live, 0, 299), "budget: capture A");
    const auto kept = store.best_match(a, 300);
    if (!kept) { check(false, "budget: A can be kept"); return; }
    auto b = a;
    b.back() = 8888;
    store.will_write(live[4]);
    cuda(cudaMemsetAsync(buffers.global + live[4] * kPage, 0x44, kPage, buffers.stream));
    check(store.capture(b, live, 0, 299), "budget: two deltas fit without evicting A");
    auto c = b;
    c.back() = 9999;
    store.will_write(live[4]);
    cuda(cudaMemsetAsync(buffers.global + live[4] * kPage, 0x55, kPage, buffers.stream));
    check(store.capture(c, live, 0, 299, kept->id), "budget: C replaces B, not kept A");
    const auto old_b = store.best_match(b, 300);
    const auto kept_after = store.best_match(a, 300);
    check(kept_after && kept_after->id == kept->id && old_b && old_b->reuse == 299 &&
              store.stats().drops == 1 && store.stats().host_bytes_used == budget &&
              store.stats().host_bytes_peak <= budget,
          "eviction reclaims B's ring/tail but retains shared pages and kept A");
    for (const int page : live) { store.release(page); }
    live.clear();
    (void)store.restore(*kept, live);
    check(tagged_page(buffers, live[4], 0x24) && tagged_ring(buffers, 0x70, 0, 5),
          "kept restore source survives shared-payload eviction byte for byte");
    for (const int page : live) { store.release(page); }

    // Recycling every physical page must not mistake old weak references for new contents.
    for (int turn = 0; turn < 12; ++turn) {
        capture_sequence(store, buffers, 10000 + turn * 1000,
                         static_cast<std::uint8_t>(0x10 + turn));
        check(store.stats().host_bytes_peak <= budget, "repeated eviction stays within budget");
    }
    const auto latest = store.best_match(prompt_after(21000), 300);
    check(latest.has_value(), "recycled physical pages retain the newest checkpoint");
    if (latest) {
        live.clear();
        (void)store.restore(*latest, live);
        check(tagged_page(buffers, live[0], 0x1b),
              "recycled page allocation receives a fresh identity");
        for (const int page : live) { store.release(page); }
    }

    PrefixStore tight(host_config(buffers, point_bytes()));
    live = tagged_pages(tight, buffers, 5, 0x30);
    check(tight.capture(a, live, 0, 299), "tight budget captures its first point");
    const auto only = tight.best_match(a, 300);
    if (only) {
        check(!tight.capture(c, live, 0, 299, only->id) &&
                  tight.stats().host_bytes_used == point_bytes() &&
                  tight.stats().captures == 1 && tight.best_match(a, 300)->id == only->id,
              "failed kept-source capture releases temporary refs/ranges without dropping source");
    }
    for (const int page : live) { tight.release(page); }
}

void wrapped_ring_coverage() {
    Buffers buffers;
    PrefixStore store(host_config(buffers, 4 * point_bytes()));
    auto live = tagged_pages(store, buffers, 16, 0x10);
    auto keys = keys_from(3000, 1000);
    cuda(cudaMemsetAsync(buffers.ring, 0x88, kRingPages * kPage, buffers.stream));
    check(store.capture(keys, live, 488, 999), "wrapped ring checkpoint captures six pages");
    check(!store.best_match(keys, 702) && store.best_match(keys, 703).has_value(),
          "wrapped coverage rejects a divergence before the retained window");
    check(!store.capture(std::vector<PositionKey>(keys.begin(), keys.begin() + 700), live, 900, 999),
          "overwritten earlier explicit boundary is not captured");
    const auto match = store.best_match(keys, 1000);
    for (const int page : live) { store.release(page); }
    live.clear();
    cuda(cudaMemsetAsync(buffers.ring, 0, kRingPages * kPage, buffers.stream));
    if (match) {
        check(store.restore(*match, live) == 640 && tagged_ring(buffers, 0x88, 10, 6),
              "ring restore preserves logical wrap and exact valid_from");
        for (const int page : live) { store.release(page); }
    }
}

void cache_write_invalidation(bool delta, bool device) {
    Buffers buffers;
    ninfer::DeviceBuffer table(kPoolPages * sizeof(std::int32_t));
    ninfer::models::gemma4::PrefixCache::Config config;
    config.store = host_config(buffers, 6 * point_bytes());
    config.store.device_slots = device ? 1 : 0;
    config.table_pages = kPoolPages;
    config.table = static_cast<std::int32_t*>(table.p);
    ninfer::models::gemma4::PrefixCache cache(std::move(config));
    const auto physical = [&](int logical) {
        std::vector<std::int32_t> pages(kPoolPages);
        cuda(cudaStreamSynchronize(buffers.stream));
        cuda(cudaMemcpy(pages.data(), table.p, table.bytes, cudaMemcpyDeviceToHost));
        return pages[static_cast<std::size_t>(logical)];
    };
    auto keys = keys_from(4000, 300);
    cache.prepare_write(0, 300);
    for (int page = 0; page < 5; ++page) {
        cuda(cudaMemsetAsync(buffers.global + physical(page) * kPage, 0x22, kPage, buffers.stream));
    }
    cuda(cudaMemsetAsync(buffers.ring, 0x33, kRingPages * kPage, buffers.stream));
    cache.wrote(0, keys);
    cache.capture(256); // Explicit boundary; first four immutable pages.
    cache.capture(300);
    const auto before = cache.stats();
    const int old_tail = physical(4);
    cache.prepare_write(299, 1); // Also covers a prepared but cancelled write.
    cache.truncate(299);
    cache.prepare_write(299, 1);
    const int new_tail = physical(4);
    check(!device || old_tail != new_tail, "device-owned partial page is copied on write");
    check(tagged_page(buffers, new_tail, 0x22), "COW/cancel preparation preserves partial bytes");
    cuda(cudaMemsetAsync(buffers.global + new_tail * kPage + 172, 0x77, 4, buffers.stream));
    const PositionKey branch = 99999;
    cache.wrote(299, {&branch, 1});
    cache.capture(300);
    const auto after = cache.stats();
    check(!delta || after.host_global_copied_bytes == before.host_global_copied_bytes + kPage,
          "exclusive and COW partial writes invalidate host page identity before capture");
    check(!delta || after.host_global_reused_bytes == before.host_global_reused_bytes + 4 * kPage,
          "untouched complete pages remain shareable across a cancelled preparation");
    check(after.captures == 3, "branch capture is retained, not skipped by the experiment");
}

void multiple_tensor_layout() {
    Buffers buffers;
    ninfer::DeviceBuffer global2(kPoolPages * (kPage / 2));
    ninfer::DeviceBuffer ring2(kRingPages * (kPage / 2));
    auto config = host_config(buffers, 6 * point_bytes());
    config.global.push_back({static_cast<std::uint8_t*>(global2.p), kPage / 2});
    config.ring.push_back({static_cast<std::uint8_t*>(ring2.p), kPage / 2});
    PrefixStore store(std::move(config));
    auto live = tagged_pages(store, buffers, 5, 0x10);
    for (int page = 0; page < 5; ++page) {
        cuda(cudaMemsetAsync(static_cast<std::uint8_t*>(global2.p) + live[page] * (kPage / 2),
                             0x90 + page, kPage / 2, buffers.stream));
    }
    cuda(cudaMemsetAsync(buffers.ring, 0x44, kRingPages * kPage, buffers.stream));
    cuda(cudaMemsetAsync(ring2.p, 0x55, kRingPages * (kPage / 2), buffers.stream));
    const auto keys = keys_from(5000, 300);
    check(store.capture(keys, live, 0, 299), "unequal global/ring tensor sizes capture");
    const auto match = store.best_match(keys, 300);
    for (const int page : live) { store.release(page); }
    live.clear();
    cuda(cudaMemsetAsync(buffers.global, 0, kPoolPages * kPage, buffers.stream));
    cuda(cudaMemsetAsync(global2.p, 0, global2.bytes, buffers.stream));
    cuda(cudaMemsetAsync(buffers.ring, 0, kRingPages * kPage, buffers.stream));
    cuda(cudaMemsetAsync(ring2.p, 0, ring2.bytes, buffers.stream));
    if (!match) { check(false, "multiple tensors match"); return; }
    (void)store.restore(*match, live);
    cuda(cudaStreamSynchronize(buffers.stream));
    bool same = true;
    std::vector<std::uint8_t> bytes(kPage / 2);
    for (int page = 0; page < 5; ++page) {
        cuda(cudaMemcpy(bytes.data(), static_cast<std::uint8_t*>(global2.p) +
                                          live[page] * (kPage / 2),
                        kPage / 2, cudaMemcpyDeviceToHost));
        same = same && tagged_page(buffers, live[page], static_cast<std::uint8_t>(0x10 + page)) &&
               std::all_of(bytes.begin(), bytes.end(), [page](auto v) { return v == 0x90 + page; });
    }
    cuda(cudaMemcpy(bytes.data(), ring2.p, bytes.size(), cudaMemcpyDeviceToHost));
    check(same && std::all_of(bytes.begin(), bytes.end(), [](auto v) { return v == 0x55; }) &&
              tagged_ring(buffers, 0x44, 0, 5),
          "page-major host payload offsets preserve every tensor independently");
    for (const int page : live) { store.release(page); }
}

void no_host_budget() {
    Buffers buffers;
    PrefixStore store(host_config(buffers, 0));
    auto live = tagged_pages(store, buffers, 5, 0x10);
    check(!store.capture(keys_from(6000, 300), live, 0, 299) &&
              store.stats().captures == 0 && store.stats().host_bytes_used == 0,
          "zero host budget does not allocate a snapshot or retain pages");
    for (const int page : live) { store.release(page); }
    live.clear();
    for (int page = 0; page < kPoolPages; ++page) { live.push_back(store.allocate_page()); }
    bool exhausted = false;
    try { (void)store.allocate_page(); }
    catch (const std::runtime_error&) { exhausted = true; }
    check(exhausted, "failed capture returns all pool capacity and allocation refuses overflow");
    for (const int page : live) { store.release(page); }
}

void pool_pressure_restore() {
    Buffers buffers;
    auto config = host_config(buffers, point_bytes());
    config.pool_pages = 10;
    config.device_slots = 2;
    PrefixStore store(std::move(config));
    capture_sequence(store, buffers, 7000, 0x10);
    capture_sequence(store, buffers, 8000, 0x20);
    std::vector<int> other_live;
    for (int page = 0; page < 5; ++page) { other_live.push_back(store.allocate_page()); }
    const auto match = store.best_match(prompt_after(7000), 300);
    check(match && match->on_host && store.stats().demotions == 1,
          "pool pressure demotes a device point without losing its ownership");
    if (match) {
        std::vector<int> restored;
        // There is no free page. Demoting B cannot evict the pinned host restore source A,
        // so B is dropped and its pool references are released instead.
        (void)store.restore(*match, restored);
        check(store.stats().drops == 1 && store.stats().host_bytes_used == point_bytes() &&
                  tagged_page(buffers, restored[0], 0x10) &&
                  !store.best_match(prompt_after(8000), 300),
              "restore allocation pressure protects its host source and frees device refs");
        for (const int page : restored) { store.release(page); }
    }
    for (const int page : other_live) { store.release(page); }
}

} // namespace

int main(int argc, char** argv) {
    try {
        // Exercise the selected startup policy through actual capture bytes, budget and
        // independent branch restoration before the explicit in-process A/B cases.
        host_delta_branches(!(argc == 2 && std::string(argv[1]) == "--expect-legacy"));
        const bool saved = ninfer::runtime_switches().gemma4_host_delta;
        for (const bool delta : {false, true}) {
            ninfer::RuntimeSwitchFlags flags;
            flags.gemma4_host_delta = delta;
            ninfer::apply_runtime_switch_flags(flags);
            restores_kept_checkpoint();
            drops_without_keep();
            touch_keeps_used_checkpoint();
            host_delta_branches(delta);
            wrapped_ring_coverage();
            cache_write_invalidation(delta, false);
            cache_write_invalidation(delta, true);
            multiple_tensor_layout();
            no_host_budget();
            pool_pressure_restore();
            if (delta) { delta_budget_and_keep(); }
        }
        ninfer::RuntimeSwitchFlags restore;
        restore.gemma4_host_delta = saved;
        ninfer::apply_runtime_switch_flags(restore);
    } catch (const std::exception& error) {
        std::cerr << "EXC " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " gemma4 prefix store\n";
    return failures ? 1 : 0;
}
