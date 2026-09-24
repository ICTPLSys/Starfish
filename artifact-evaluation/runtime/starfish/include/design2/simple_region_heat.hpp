#pragma once

// Counts demand references to physical Region slots. No object census,
// lifetime or quota accounting. Optional two-class labels use the same ranking.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "design2/simple_region_budget.hpp"
#include "design2/simple_dirty_observer.hpp"

namespace FarLib::simple_region_heat {

inline constexpr uint8_t kCold = 1, kHot = 4; // dirty dimension fixed Medium
inline bool grouping_enabled() {
    static const bool on = [] {
        const char *p = std::getenv("FARLIB_SIMPLE_HOTCOLD");
        return p && std::strcmp(p, "1") == 0;
    }();
    return on;
}
inline size_t class_count() { return simple_region_budget::classes(); }
inline bool is_hot(uint32_t c) { return simple_region_budget::six_enabled() ? c>=3 && c<6 : c==kHot; }
inline uint8_t normalize_class(uint32_t c) { return simple_region_budget::six_enabled() ? uint8_t(c<6?c:kCold) : (c == kHot ? kHot : kCold); }
inline uint8_t opposite_class(uint32_t c) { c=normalize_class(c); return simple_region_budget::six_enabled() ? uint8_t(c<3?c+3:c-3) : (c==kHot?kCold:kHot); }
inline std::array<uint8_t,6> fallback_order(uint32_t requested) {
    std::array<uint8_t,6> result{}; result[0]=normalize_class(requested);result[1]=opposite_class(requested);
    size_t n=2; if(simple_region_budget::six_enabled()) for(uint8_t c=0;c<6;++c) if(c!=result[0] && c!=result[1]) result[n++]=c;
    return result;
}

inline bool relink_enabled() {
    static const bool on = [] {
        const char *p = std::getenv("FARLIB_SIMPLE_HOTCOLD_RELINK");
        return !(p && std::strcmp(p, "0") == 0);
    }();
    return grouping_enabled() && on;
}

inline bool local_routing_enabled() {
    static const bool on = [] {
        const char *p = std::getenv("FARLIB_SIMPLE_LOCAL_ROUTING");
        return !(p && std::strcmp(p, "0") == 0);
    }();
    return grouping_enabled() && on;
}

// Independent ablation: local labels/lists remain enabled when this is off.
// Remote grouping is opt-in; the default is the original unified allocator.
inline bool remote_grouping_enabled() {
    static const bool on = [] {
        const char *p = std::getenv("FARLIB_SIMPLE_REMOTE_HOTCOLD");
        return p && std::strcmp(p, "1") == 0;
    }();
    return grouping_enabled() && on;
}
inline uint8_t fetch_class_hint(uint32_t source_class, uint32_t remote_class) {
    return normalize_class(remote_grouping_enabled() ? remote_class : source_class);
}

// Physical-slot labels only. No object registry, quota or lifetime accounting.
class Classes {
    uintptr_t base_ = 0;
    size_t bytes_ = 0, region_bytes_ = 1, slots_ = 0;
    std::unique_ptr<std::atomic<uint8_t>[]> labels_;
    // Allocation/supply labels are deliberately independent from the heat
    // labels above.  Heat windows may relabel a live physical Region, while
    // the soft budget owns the supply class until an empty Region is
    // detached and reinitialized.  The old bin is needed to transfer that
    // ownership without treating a bin recycle as a live migration.
    std::unique_ptr<std::atomic<uint8_t>[]> supply_labels_;
    std::unique_ptr<std::atomic<int>[]> supply_bins_;
public:
    void configure(uintptr_t base, size_t bytes, size_t region_bytes) {
        if(!region_bytes)throw std::invalid_argument("zero Region size");
        // Cache setup may call monitor configuration more than once while
        // wiring callbacks.  Do not erase supply ownership after claims have
        // already been made for this exact physical heap.
        if (simple_region_budget::enabled() && labels_ &&
            base_ == base && bytes_ == bytes &&
            region_bytes_ == region_bytes)
            return;
        base_=base; bytes_=bytes; region_bytes_=region_bytes;
        slots_=(bytes+region_bytes-1)/region_bytes;
        labels_=std::make_unique<std::atomic<uint8_t>[]>(slots_);
        if (simple_region_budget::enabled()) {
            supply_labels_=std::make_unique<std::atomic<uint8_t>[]>(slots_);
            supply_bins_=std::make_unique<std::atomic<int>[]>(slots_);
        } else {
            supply_labels_.reset();
            supply_bins_.reset();
        }
        for(size_t i=0;i<slots_;++i) {
            labels_[i].store(kCold);
            if (simple_region_budget::enabled()) {
                supply_labels_[i].store(kCold);
                supply_bins_[i].store(-1);
            }
        }
    }
    uint8_t get(uintptr_t address) const {
        if(!labels_ || address<base_ || address-base_>=bytes_)return kCold;
        return labels_[(address-base_)/region_bytes_].load(std::memory_order_relaxed);
    }
    void set(uintptr_t address, uint8_t c) {
        if(labels_ && address>=base_ && address-base_<bytes_)
            labels_[(address-base_)/region_bytes_].store(normalize_class(c),std::memory_order_relaxed);
    }
    uint8_t allocation_class(uintptr_t address) const {
        if(!labels_ || address<base_ || address-base_>=bytes_)return kCold;
        const size_t slot = (address-base_)/region_bytes_;
        if (!simple_region_budget::enabled())
            return labels_[slot].load(std::memory_order_relaxed);
        if (!supply_labels_) return kCold;
        return supply_labels_[slot].load(std::memory_order_relaxed);
    }
    uint8_t assign_supply_empty(uintptr_t address, size_t bin,
                                uint8_t requested) {
        requested = normalize_class(requested);
        // The source/object hint remains a heat label even when the physical
        // supply class is budget-owned and stable across heat windows.
        set(address, requested);
        if (!simple_region_budget::enabled()) {
            return requested;
        }

        int old_bin = -1;
        uint8_t old_class = kCold;
        size_t slot = 0;
        const bool mapped = labels_ && address>=base_ && address-base_<bytes_;
        if (!mapped || !supply_labels_ || !supply_bins_)
            throw std::logic_error(
                "simple region budget local heap classes not configured");
        if (mapped) {
            slot = (address-base_)/region_bytes_;
            old_bin = supply_bins_[slot].load(std::memory_order_relaxed);
            old_class = supply_labels_[slot].load(std::memory_order_relaxed);
        }
        const uint8_t chosen = simple_region_budget::local().claim(
            bin, requested, old_bin, old_class);
        if (mapped) {
            supply_labels_[slot].store(chosen, std::memory_order_relaxed);
            supply_bins_[slot].store(static_cast<int>(bin),
                                     std::memory_order_relaxed);
        }
        return chosen;
    }
    uint8_t reassign_supply_public(uintptr_t address, size_t bin,
                                   uint8_t old_class) {
        old_class = normalize_class(old_class);
        if (!simple_region_budget::enabled()) return old_class;
        const bool mapped = labels_ && address>=base_ && address-base_<bytes_;
        if (!mapped || !supply_labels_ || !supply_bins_)
            throw std::logic_error(
                "simple region budget local heap classes not configured");
        const size_t slot = (address-base_)/region_bytes_;
        const int old_bin = supply_bins_[slot].load(std::memory_order_relaxed);
        if (old_bin != static_cast<int>(bin))
            throw std::logic_error(
                "simple region budget background bin mismatch");
        const uint8_t current = normalize_class(
            supply_labels_[slot].load(std::memory_order_relaxed));
        if (current != old_class) old_class = current;
        const uint8_t chosen = simple_region_budget::local().reassign_public(
            bin, old_class);
        supply_labels_[slot].store(chosen, std::memory_order_relaxed);
        supply_bins_[slot].store(static_cast<int>(bin),
                                 std::memory_order_relaxed);
        return chosen;
    }
    size_t slots() const { return slots_; }
    uint64_t epoch() const { return epoch_.load(std::memory_order_acquire); }
    size_t apply(const std::vector<struct Rank> &rows);
private:
    std::atomic<uint64_t> epoch_{1};
};
inline Classes &classes() { static Classes c; return c; }
inline uint8_t class_for(uintptr_t address) { return classes().get(address); }
inline uint8_t allocation_class_for(uintptr_t address) {
    return classes().allocation_class(address);
}
inline uint8_t assign_supply_empty(uintptr_t address, size_t bin,
                                   uint8_t requested) {
    return classes().assign_supply_empty(address, bin, requested);
}
inline uint8_t reassign_supply_public(uintptr_t address, size_t bin,
                                      uint8_t old_class) {
    return classes().reassign_supply_public(address, bin, old_class);
}

inline bool enabled() {
    static const bool on = [] {
        const char *p = std::getenv("FARLIB_SIMPLE_REGION_HEAT");
        return p && std::strcmp(p, "1") == 0;
    }();
    return on;
}

inline uint64_t interval_ms() {
    static const uint64_t value = [] {
        const char *p = std::getenv("FARLIB_SIMPLE_REGION_HEAT_INTERVAL_MS");
        if (p == nullptr || *p == '\0') return uint64_t{10000};
        char *end = nullptr;
        const auto parsed = std::strtoull(p, &end, 10);
        if (end == p || parsed == 0) return uint64_t{10000};
        return std::min<uint64_t>(parsed, 60000);
    }();
    return value;
}

// Capacity feedback must not shorten the independent heat evidence window.
inline uint64_t budget_interval_ms() {
    static const uint64_t value = [] {
        const char *p = std::getenv("FARLIB_SIMPLE_REGION_BUDGET_INTERVAL_MS");
        if (p == nullptr || *p == '\0') return interval_ms();
        char *end = nullptr;
        const auto parsed = std::strtoull(p, &end, 10);
        if (end == p || *end != '\0' || parsed == 0 || parsed > 60000)
            throw std::invalid_argument("invalid Region budget interval");
        return uint64_t(parsed);
    }();
    return value;
}

struct Sampler {
    uint64_t sequence = 0;
    bool select() { return (++sequence & 63U) == 0; }
};

struct Rank {
    size_t region;
    uint64_t samples;
};

inline size_t Classes::apply(const std::vector<Rank> &rows) {
    // Include zero-count slots; ties use ascending physical Region index.
    const size_t hot_count = slots_ / 5 * 4 + (slots_ % 5) * 4 / 5;
    std::vector<uint8_t> next(slots_, kCold), seen(slots_, 0);
    size_t selected=0;
    for(const auto &r:rows) {
        if(r.region>=slots_ || seen[r.region])throw std::logic_error("invalid heat ranking");
        seen[r.region]=1;
        if(selected<hot_count) { next[r.region]=kHot; ++selected; }
    }
    for(size_t i=0;i<slots_ && selected<hot_count;++i)
        if(!seen[i]) { next[i]=kHot; ++selected; }
    size_t changed=0;
    for(size_t i=0;i<slots_;++i)
        changed += labels_[i].exchange(next[i],std::memory_order_relaxed)!=next[i];
    epoch_.fetch_add(1, std::memory_order_release);
    return changed;
}

class Counters {
    uintptr_t local_base_ = 0;
    size_t local_bytes_ = 0, region_bytes_ = 1, slots_ = 0;
    std::unique_ptr<std::atomic<uint64_t>[]> counts_;
public:
    void configure(uintptr_t base, size_t local_bytes, size_t region_bytes) {
        if (!region_bytes)
            throw std::invalid_argument("zero Region size");
        local_base_ = base; local_bytes_ = local_bytes;
        region_bytes_ = region_bytes;
        slots_ = (local_bytes + region_bytes - 1) / region_bytes;
        counts_ = std::make_unique<std::atomic<uint64_t>[]>(slots_);
        for (size_t i=0;i<slots_;++i) counts_[i].store(0, std::memory_order_relaxed);
    }
    bool add(uint64_t address) {
        if (address < local_base_ || address - local_base_ >= local_bytes_) return false;
        const size_t slot = (address - local_base_) / region_bytes_;
        counts_[slot].fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    std::vector<Rank> take(bool sort = true) {
        std::vector<Rank> result;
        for (size_t i=0;i<slots_;++i) {
            const auto n = counts_[i].exchange(0, std::memory_order_relaxed);
            if (n) result.push_back({i, n});
        }
        if (sort) std::sort(result.begin(),result.end(),[](const Rank &a,const Rank &b) {
            if (a.samples != b.samples) return a.samples > b.samples;
            return a.region < b.region;
        });
        return result;
    }
    size_t slots() const { return slots_; }
};

class Monitor {
    Counters counters_;
    std::atomic_bool active_{false};
    std::atomic<uint64_t> unmapped_{0};
    std::mutex mutex_;
    std::condition_variable wake_;
    bool stop_ = false;
    std::thread worker_;
    std::ofstream output_;
    uint64_t window_ = 0;
    std::function<size_t()> reclassify_;
    std::function<void(uint64_t)> budget_tick_;
    using Clock = std::chrono::steady_clock;
    void publish(const std::vector<Rank> &rows, bool partial, uint64_t elapsed_ns,
                 uint64_t collection_ns) {
        uint64_t sum=0;for(const auto &r:rows) sum+=r.samples;
        const auto unmapped=unmapped_.exchange(0,std::memory_order_relaxed);
        if (!partial && output_) {
            size_t rank=0;
            for(const auto &r:rows)
                output_ << window_ << '\t' << ++rank << '\t'
                        << r.region << '\t' << r.samples << '\n';
            output_.flush();
        }
        std::cout << "simple_region_heat.window index=" << window_
                  << " partial=" << partial << " elapsed_ns=" << elapsed_ns
                  << " sampled=" << sum << " unmapped=" << unmapped
                  << " nonzero_regions=" << rows.size()
                  << " slots_scanned=" << counters_.slots()
                  << " collect_sort_ns=" << collection_ns << std::endl;
    }
public:
    ~Monitor() { end(); }
    void configure(uintptr_t base, size_t local_bytes, size_t region_bytes) {
        if (worker_.joinable()) throw std::logic_error("configure active heat monitor");
        if (simple_region_budget::six_enabled() && !simple_region_budget::enabled())
            throw std::invalid_argument("semantic six groups require Region budgets");
        if (simple_region_budget::enabled() &&
            (!grouping_enabled() || !local_routing_enabled() || !remote_grouping_enabled()))
            throw std::invalid_argument("Region budgets require local and remote Hot/Cold routing");
        if (simple_region_budget::enabled())
            std::cerr << "simple_region_budget.config mode="
                      << (simple_region_budget::mode() == simple_region_budget::Mode::Adaptive ? "adaptive" : "fixed")
                      << " interval_ms=" << budget_interval_ms()
                      << " heat_interval_ms=" << interval_ms()
                      << " fast=" << simple_region_budget::fast_enabled()
                      << " min_requests=32 shortage_pct=10 donor_max_miss_pct=5 consecutive_windows=2"
                      << " severe_initial_windows=" << (simple_region_budget::fast_enabled() ? 1 : 2)
                      << " multi_receiver=" << simple_region_budget::fast_enabled()
                      << " max_step_pct=" << (simple_region_budget::fast_enabled() ? 10 : 1)
                      << " minimum_step_regions=1 live_migration=0 private_steal=0"
                      << " heat_and_supply_separate=1 scope=domain_bin"
                      << " actuator=background_public_usable semantic_classes=" << simple_region_budget::classes() << std::endl;
        for(const char *name:{"FARLIB_FIXED_SIX_GROUPS","FARLIB_FIXED_SIX_POOLS"}) {
            const char *v=std::getenv(name);
            if(v && std::strcmp(v,"0")!=0)
                throw std::invalid_argument("simple heat must not run with old grouping");
        }
        counters_.configure(base,local_bytes,region_bytes);
        if(grouping_enabled()) {
            const char *lists=std::getenv("FARLIB_LIST_ONLY_SIX");
            if(!lists || std::strcmp(lists,"1")!=0)
                throw std::invalid_argument("hotcold requires six child lists");
            classes().configure(base,local_bytes,region_bytes);
            std::cerr << "simple_hotcold.config enabled=1 hot_percent=80 cold_class=1 hot_class=4 quota=0 denominator=physical_local_slots" << std::endl;
            std::cerr << "simple_remote_hotcold.config enabled=" << remote_grouping_enabled()
                      << " local_hint_preserved=1 relink_enabled=" << relink_enabled()
                      << " local_routing_enabled=" << local_routing_enabled() << std::endl;
        }
        if(const char *path=std::getenv("FARLIB_SIMPLE_REGION_HEAT_PATH")) {
            output_.open(path);
            if(!output_)throw std::runtime_error("cannot open simple heat output");
            output_ << "window\trank\tphysical_region\tsampled_accesses\n";
        }
        std::cerr << "simple_region_heat.config sample_period=64 interval_ms="
                  << interval_ms()
                  << " metric=access_count migration=0 quota=0 slots="
                  << counters_.slots() << std::endl;
    }
    void set_reclassify(std::function<size_t()> fn) { reclassify_=std::move(fn); }
    void set_budget_tick(std::function<void(uint64_t)> fn) { budget_tick_=std::move(fn); }
    bool select() {
        if(!active_.load(std::memory_order_relaxed))return false;
        static thread_local Sampler sampler;
        return sampler.select();
    }
    void add(uint64_t address) {
        if(!counters_.add(address))unmapped_.fetch_add(1,std::memory_order_relaxed);
    }
    void begin() {
        if(!counters_.slots())throw std::logic_error("heat monitor not configured");
        if(worker_.joinable())throw std::logic_error("heat monitor already running");
        if (simple_region_budget::enabled()) {
            simple_region_budget::local().begin();
            simple_region_budget::remote().begin();
        }
        if (simple_dirty_observer::enabled()) simple_dirty_observer::monitor().begin();
        stop_=false;window_=0;active_.store(true,std::memory_order_relaxed);
        worker_=std::thread([this] {
            auto start=Clock::now();
            const auto origin=start;
            auto budget_start=start;
            uint64_t budget_window=0;
            const bool has_budget=simple_region_budget::enabled() && bool(budget_tick_);
            const auto heat_period=std::chrono::milliseconds(interval_ms());
            const auto budget_period=std::chrono::milliseconds(budget_interval_ms());
            std::unique_lock lock(mutex_);
            for (;;) {
                const auto deadline=has_budget ? std::min(start+heat_period,budget_start+budget_period) : start+heat_period;
                if (wake_.wait_until(lock,deadline,[this]{return stop_;})) break;
                lock.unlock();
                const auto now=Clock::now();
                if (now >= start+heat_period) {
                const auto before=Clock::now();auto rows=counters_.take();
                const auto after=Clock::now();++window_;
                if(grouping_enabled()) {
                    const size_t changed=classes().apply(rows);
                    const size_t moved=(relink_enabled() && reclassify_) ? reclassify_() : 0;
                    const auto finished=Clock::now();
                    const size_t hot=classes().slots()/5*4+(classes().slots()%5)*4/5;
                    std::cout << "simple_hotcold.window index=" << window_
                              << " hot=" << hot << " cold=" << classes().slots()-hot
                              << " labels_changed=" << changed << " shared_relinked=" << moved
                              << " relink_enabled=" << relink_enabled()
                              << " local_routing_enabled=" << local_routing_enabled()
                              << " apply_relink_ns=" << std::chrono::duration_cast<std::chrono::nanoseconds>(finished-after).count()
                              << std::endl;
                } else if (reclassify_) {
                    // The six-independent-list diagnostic uses the same
                    // ten-second boundary but does not enable Hot/Cold
                    // classification or relinking.
                    reclassify_();
                }
                publish(rows,false,std::chrono::duration_cast<std::chrono::nanoseconds>(before-start).count(),
                        std::chrono::duration_cast<std::chrono::nanoseconds>(after-before).count());
                if (simple_dirty_observer::enabled())
                    simple_dirty_observer::monitor().snapshot(window_,false);
                start=before;
                }
                if (has_budget && now >= budget_start+budget_period) {
                    ++budget_window;
                    std::cout << "simple_region_budget.clock window=" << budget_window
                              << " elapsed_ns=" << std::chrono::duration_cast<std::chrono::nanoseconds>(now-budget_start).count()
                              << " since_begin_ns=" << std::chrono::duration_cast<std::chrono::nanoseconds>(now-origin).count()
                              << std::endl;
                    budget_tick_(budget_window);
                    budget_start=now;
                }
                lock.lock();
            }
            lock.unlock();
            const auto before=Clock::now();auto tail=counters_.take(false);
            const auto after=Clock::now();
            publish(tail,true,std::chrono::duration_cast<std::chrono::nanoseconds>(before-start).count(),
                    std::chrono::duration_cast<std::chrono::nanoseconds>(after-before).count());
        });
    }
    void end() {
        if(!worker_.joinable())return;
        active_.store(false,std::memory_order_relaxed);
        {std::lock_guard lock(mutex_);stop_=true;}
        wake_.notify_all();worker_.join();
        if (simple_dirty_observer::enabled()) simple_dirty_observer::monitor().end();
        if (simple_region_budget::enabled()) {
            simple_region_budget::local().end();
            simple_region_budget::remote().end();
        }
    }
};

inline Monitor &monitor() { static Monitor m; return m; }
inline void begin_work() { if(enabled())monitor().begin(); }
inline void end_work() { if(enabled())monitor().end(); }

} // namespace FarLib::simple_region_heat
