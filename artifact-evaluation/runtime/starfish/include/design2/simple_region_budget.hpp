#pragma once

// Small, opt-in capacity feedback baseline. Heat evidence and allocation
// capacity labels are deliberately separate. No per-object hooks or migration.
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <algorithm>

namespace FarLib::simple_region_budget {
inline constexpr size_t kBins = 56;
inline constexpr size_t kClasses = 6;
inline bool six_enabled() {
    static const bool on = [] { const char *s = std::getenv("FARLIB_SIMPLE_SIX_GROUPS"); return s && std::strcmp(s,"1")==0; }();
    return on;
}
// The fast controller is deliberately scoped to the semantic six-group
// policy.  A fast flag in a legacy two-group process must be inert so that
// existing allocation and cadence behavior remains unchanged.
inline bool fast_enabled() {
    static const bool on = [] {
        const char *s = std::getenv("FARLIB_SIMPLE_REGION_BUDGET_FAST");
        return six_enabled() && s && std::strcmp(s, "1") == 0;
    }();
    return on;
}
inline size_t classes() { return six_enabled() ? 6 : 2; }
inline constexpr uint8_t kCold = 1, kHot = 4;
using Counts = std::array<uint64_t, kClasses>;
using Supply = std::array<Counts, kBins>;
inline size_t index(uint8_t label) { return six_enabled() ? (label<6?label:kCold) : (label == kHot ? 1 : 0); }
inline uint8_t label(size_t i) { return six_enabled() ? uint8_t(i) : (i == 1 ? kHot : kCold); }
inline uint64_t sum(const Counts &a) { uint64_t n=0; for(auto v:a)n+=v;return n; }
enum class Mode { Off, Fixed, Adaptive };
inline Mode mode() {
    static const Mode value = [] {
        const char *s = std::getenv("FARLIB_SIMPLE_REGION_BUDGET");
        if (!s || !*s || std::strcmp(s, "0") == 0) return Mode::Off;
        if (std::strcmp(s, "fixed") == 0) return Mode::Fixed;
        if (std::strcmp(s, "adaptive") == 0) return Mode::Adaptive;
        throw std::invalid_argument("FARLIB_SIMPLE_REGION_BUDGET must be 0, fixed or adaptive");
    }();
    return value;
}
inline bool enabled() { return mode() != Mode::Off; }

struct Snapshot {
    Counts actual{}, target{}, attempts{}, misses{}, free_bytes{};
    uint64_t transferred = 0, applied = 0;
    unsigned streak = 0;
};

class Controller {
    struct Bin {
        Counts actual{}, target{};
        // One atomic exchange captures numerator and denominator together.
        // A window is expected to contain < 2^32 global Region claims/class.
        std::array<std::atomic<uint64_t>, kClasses> requests{};
        int direction = -1;
        unsigned streak = 0;
        // Fast six-group role evidence.  0 is no current role, 1 is a
        // receiver, and 2 is a donor.  Keeping this separate from the
        // legacy receiver streak preserves the old two-group policy exactly.
        std::array<uint8_t, kClasses> fast_role{};
        std::array<uint8_t, kClasses> fast_role_streak{};
        // Last role that actually participated in a planned transfer.  This
        // survives an idle/no-demand tick so a quiet-then-reverse class still
        // pays the two-tick reversal guard without carrying stale evidence.
        std::array<uint8_t, kClasses> fast_last_transfer_role{};
        uint64_t applied = 0;
        Snapshot last{};
    };
    const char *name_;
    Mode mode_;
    mutable std::mutex mutex_;
    std::array<Bin, kBins> bins_{};
    size_t region_bytes_ = 0;
    bool initialized_ = false;
    std::atomic<bool> active_{false};
    static void check_bin(size_t bin) {
        if (bin >= kBins) throw std::out_of_range("Region budget bin");
    }

    // Split a bounded Region budget over eligible classes.  When the budget
    // can cover every eligible class, reserve one Region for each first; the
    // remainder is allocated proportionally and completed with deterministic
    // largest-remainder tie breaking.  limits is used for donors and is
    // UINT64_MAX for unconstrained receivers.
    static Counts fast_split(const Counts &weights,
                             const std::array<bool, kClasses> &eligible,
                             const Counts &limits,
                             uint64_t budget) {
        Counts result{};
        std::array<long double, kClasses> fractions{};
        size_t count = 0;
        uint64_t capacity = 0;
        uint64_t weight_sum = 0;
        for (size_t i = 0; i < kClasses; ++i) {
            if (!eligible[i]) continue;
            ++count;
            capacity = (limits[i] > UINT64_MAX - capacity)
                           ? UINT64_MAX
                           : capacity + limits[i];
            weight_sum = (weights[i] > UINT64_MAX - weight_sum)
                             ? UINT64_MAX
                             : weight_sum + weights[i];
        }
        if (!count || !budget) return result;
        if (capacity < budget) budget = capacity;

        // A nonzero fair minimum is possible only when there is enough
        // budget for every receiver/donor selected by this split.
        if (budget >= count) {
            for (size_t i = 0; i < kClasses; ++i)
                if (eligible[i] && limits[i]) {
                    ++result[i];
                    --budget;
                }
        }
        if (!budget) return result;

        if (!weight_sum) {
            // This is only a defensive path (receivers have miss weights and
            // donors have positive capacities).  It remains conservation-safe.
            while (budget) {
                size_t pick = kClasses;
                for (size_t i = 0; i < kClasses; ++i)
                    if (eligible[i] && result[i] < limits[i] &&
                        (pick == kClasses || i < pick)) pick = i;
                if (pick == kClasses) break;
                const auto room = limits[pick] - result[pick];
                const auto take = std::min<uint64_t>(budget, room);
                result[pick] += take;
                budget -= take;
            }
            return result;
        }

        const uint64_t proportional_budget = budget;
        uint64_t assigned = 0;
        for (size_t i = 0; i < kClasses; ++i) {
            if (!eligible[i]) continue;
            // The request counters are bounded to 32 bits per window in the
            // packed signal, so long double gives an exact floor for the
            // practical range while avoiding a non-portable __int128.
            const long double exact =
                static_cast<long double>(proportional_budget) *
                static_cast<long double>(weights[i]) /
                static_cast<long double>(weight_sum);
            uint64_t whole = static_cast<uint64_t>(exact);
            const uint64_t room = limits[i] - result[i];
            if (whole > room) whole = room;
            result[i] += whole;
            assigned += whole;
            fractions[i] = exact - static_cast<long double>(whole);
        }
        if (assigned >= proportional_budget) return result;
        uint64_t left = proportional_budget - assigned;
        // Largest-remainder completion assigns at most one Region per class
        // in this phase.  In the usual case left < eligible_count; if limits
        // made earlier floors saturate, a bounded capacity fill below handles
        // any residual after every class has had one fair opportunity.
        while (left) {
            size_t pick = kClasses;
            for (size_t i = 0; i < kClasses; ++i) {
                if (!eligible[i] || result[i] >= limits[i]) continue;
                if (pick == kClasses || fractions[i] > fractions[pick] ||
                    (fractions[i] == fractions[pick] && i < pick))
                    pick = i;
            }
            if (pick == kClasses) break;
            result[pick] += 1;
            --left;
            // A class that consumed its fractional remainder should not win
            // another one-unit round solely on a stale remainder.
            fractions[pick] = -1.0L;
        }
        return result;
    }
public:
    explicit Controller(const char *name, Mode m) : name_(name), mode_(m) {}
    void configure(size_t region_bytes) {
        if (region_bytes == 0) throw std::invalid_argument("zero budget Region size");
        std::lock_guard lock(mutex_);
        if (active_.load()) throw std::logic_error("configure active Region budget");
        region_bytes_ = region_bytes;
        initialized_ = false;
        for (auto &b : bins_) {
            b.actual = {}; b.target = {}; b.last = {};
            b.direction = -1; b.streak = 0; b.applied = 0;
            b.fast_role = {}; b.fast_role_streak = {};
            b.fast_last_transfer_role = {};
            for (auto &r : b.requests) r.store(0, std::memory_order_relaxed);
        }
    }
    void begin() {
        if (mode_ == Mode::Off || region_bytes_ == 0) return;
        std::lock_guard lock(mutex_);
        for (auto &b : bins_) {
            b.target = b.actual;
            b.direction = -1; b.streak = 0; b.applied = 0;
            b.fast_role = {}; b.fast_role_streak = {};
            b.fast_last_transfer_role = {};
            for (auto &r : b.requests) r.store(0, std::memory_order_relaxed);
        }
        initialized_ = true;
        active_.store(true, std::memory_order_release);
    }
    void end() {
        if (!active_.exchange(false, std::memory_order_acq_rel)) return;
        std::lock_guard lock(mutex_);
        for (size_t bin = 0; bin < kBins; ++bin) {
            const auto &b = bins_[bin];
            if (sum(b.actual) == 0) continue;
            if (six_enabled()) {
                for(size_t c=0;c<classes();++c)
                    std::cout << "simple_region_budget.six_final domain=" << name_ << " bin=" << bin << " class=" << c
                              << " actual=" << b.actual[c] << " target=" << b.target[c] << " applied_total=" << b.applied << std::endl;
                continue;
            }
            std::cout << "simple_region_budget.final domain=" << name_ << " bin=" << bin
                      << " actual_cold=" << b.actual[0] << " actual_hot=" << b.actual[1]
                      << " target_cold=" << b.target[0] << " target_hot=" << b.target[1]
                      << " applied_total=" << b.applied
                      << " pending=" << (b.target[0] > b.actual[0] ? b.target[0] - b.actual[0] : b.actual[0] - b.target[0])
                      << std::endl;
        }
    }
    void note_request(size_t bin, uint8_t requested, bool missed) {
        if (!active_.load(std::memory_order_relaxed)) return;
        check_bin(bin);
        bins_[bin].requests[index(requested)].fetch_add(
            (uint64_t{1} << 32) + uint64_t(missed), std::memory_order_relaxed);
    }
    // Call only for a detached, empty descriptor whose former bin/class is
    // known, or old_bin=-1 for a newly committed descriptor. A target is soft:
    // allocation never fails because it is over target.
    uint8_t claim(size_t bin, uint8_t requested, int old_bin = -1,
                  uint8_t old_class = kCold) {
        check_bin(bin);
        std::lock_guard lock(mutex_);
        if (region_bytes_ == 0) throw std::logic_error("unconfigured Region budget");
        auto &b = bins_[bin];
        size_t chosen = index(requested);
        if (old_bin >= 0) {
            check_bin(size_t(old_bin));
            auto &old = bins_[size_t(old_bin)];
            const auto prev = index(old_class);
            if (!old.actual[prev]) throw std::logic_error("Region budget ownership underflow");
            if (size_t(old_bin) == bin && initialized_) {
                // Foreground recycling preserves the supply class. Only the
                // background public-Region actuator realizes target changes.
                return label(prev);
            }
            --old.actual[prev];
            // A cross-bin recycle changes that bin's committed envelope, not
            // a policy transfer across size bins. Keep both sums conserved.
            if (old.target[prev]) --old.target[prev];
            else for(size_t c=0;c<classes();++c) if(old.target[c]) { --old.target[c]; break; }
        }
        ++b.actual[chosen]; ++b.target[chosen];
        return label(chosen);
    }
    // A read-only selection hint, not a reservation. The actuator must detach
    // a public usable descriptor before reassign_public() rechecks the debt.
    Supply pending_transfers() const {
        Supply pending{};
        std::lock_guard lock(mutex_);
        if (mode_ != Mode::Adaptive || !active_.load(std::memory_order_acquire))
            return pending;
        for (size_t bin = 0; bin < kBins; ++bin) {
            const auto &b = bins_[bin];
            for (size_t d = 0; d < classes(); ++d)
                if (b.actual[d] > b.target[d]) pending[bin][d] = b.actual[d] - b.target[d];
        }
        return pending;
    }
    // Background-only. The caller has detached a globally public USABLE
    // descriptor with free allocation slots, in this same bin/supply class.
    // Existing live objects remain in place: this changes the allocation pool,
    // not their measured heat. Never steal a private/full Region. The caller
    // must serialize deallocation as needed and republish the unchanged header
    // to the returned class's usable list (also when no transfer is needed).
    uint8_t reassign_public(size_t bin, uint8_t old_class) {
        check_bin(bin);
        std::lock_guard lock(mutex_);
        if (mode_ != Mode::Adaptive || !active_.load(std::memory_order_acquire))
            return old_class;
        auto &b = bins_[bin];
        const size_t d = index(old_class);
        if (b.actual[d] > b.target[d])
            for(size_t r=0;r<classes();++r) if(b.actual[r]<b.target[r]) {
                --b.actual[d]; ++b.actual[r]; ++b.applied; return label(r);
            }
        return old_class;
    }
    Snapshot snapshot(size_t bin) const {
        check_bin(bin);
        std::lock_guard lock(mutex_);
        auto s = bins_[bin].last;
        s.actual = bins_[bin].actual; s.target = bins_[bin].target;
        s.applied = bins_[bin].applied;
        return s;
    }
    void tick(const Supply &supply, uint64_t window) {
        if (!active_.load(std::memory_order_acquire)) return;
        std::lock_guard lock(mutex_);
        for (size_t bin = 0; bin < kBins; ++bin) {
            auto &b = bins_[bin];
            Snapshot s;
            s.free_bytes = supply[bin];
            for (size_t i = 0; i < classes(); ++i) {
                const uint64_t r = b.requests[i].exchange(0, std::memory_order_relaxed);
                s.attempts[i] = r >> 32;
                s.misses[i] = r & 0xffffffffULL;
            }
            const uint64_t total = sum(b.actual);
            if (fast_enabled() && mode_ == Mode::Adaptive) {
                // Fast policy: six semantic classes can all participate in
                // one bounded tick.  A receiver needs >=32 attempts and a
                // >=10% miss ratio; >=50% is severe and permits an initial
                // or sustained same-role transfer without waiting.
                std::array<bool, kClasses> receivers{};
                std::array<bool, kClasses> donors{};
                std::array<bool, kClasses> ready{};
                Counts donor_limits{};
                bool any_shortage = false;
                long double max_ready_miss_fraction = 0.0L;
                int best_receiver = -1;

                for (size_t c = 0; c < classes(); ++c) {
                    const uint64_t attempts = s.attempts[c];
                    const uint64_t misses = s.misses[c];
                    const bool shortage = attempts >= 32 && misses * 10 >= attempts;
                    const bool severe = shortage && misses * 2 >= attempts;
                    const uint64_t public_regions = s.free_bytes[c] / region_bytes_;
                    const bool quiet = attempts == 0 || misses * 20 < attempts;
                    const uint64_t available =
                        b.target[c] > 1 ? b.target[c] - 1 : uint64_t{0};
                    const bool donor = quiet && total > 1 && public_regions > 0 &&
                                       available > 0;
                    receivers[c] = shortage;
                    any_shortage = any_shortage || shortage;

                    const uint8_t role = shortage ? uint8_t{1}
                                                  : (donor ? uint8_t{2} : uint8_t{0});
                    const uint8_t previous = b.fast_role[c];
                    const uint8_t last_transfer = b.fast_last_transfer_role[c];
                    if (!role) {
                        // No evidence this tick must erase both role and
                        // streak, preventing a stale role after idle demand.
                        b.fast_role[c] = 0;
                        b.fast_role_streak[c] = 0;
                    } else if (previous == role) {
                        b.fast_role_streak[c] =
                            std::min<uint8_t>(2, uint8_t(b.fast_role_streak[c] + 1));
                    } else {
                        // A role reversal starts a fresh two-tick guard.
                        b.fast_role[c] = role;
                        b.fast_role_streak[c] = 1;
                    }

                    if (shortage) {
                        // Severe evidence is immediately actionable only on
                        // initial/same-direction transfer evidence.  A donor
                        // to receiver reversal must still prove itself twice,
                        // even if an idle tick reset current-role evidence.
                        const bool reversal = last_transfer != 0 &&
                                              last_transfer != uint8_t{1};
                        ready[c] = severe ? (!reversal ||
                                             b.fast_role_streak[c] >= 2)
                                          : b.fast_role_streak[c] >= 2;
                        if (ready[c])
                            max_ready_miss_fraction =
                                std::max(max_ready_miss_fraction,
                                         static_cast<long double>(misses) /
                                             static_cast<long double>(attempts));
                        if (best_receiver < 0 ||
                            misses > s.misses[size_t(best_receiver)] ||
                            (misses == s.misses[size_t(best_receiver)] && c <
                                                                  size_t(best_receiver)))
                            best_receiver = int(c);
                    }

                    // Donor role reversals use the same guard as receivers.
                    // This is intentionally based on the last role that
                    // actually transferred, not on a stale idle streak.
                    const bool donor_reversal = last_transfer != 0 &&
                                                last_transfer != uint8_t{2};
                    const bool donor_ready = donor &&
                                             (!donor_reversal ||
                                              b.fast_role_streak[c] >= 2);
                    donors[c] = donor_ready;
                    donor_limits[c] = donor_ready
                                          ? std::min(available, public_regions)
                                          : 0;
                }

                if (best_receiver < 0) {
                    b.direction = -1;
                    b.streak = 0;
                } else {
                    b.direction = best_receiver;
                    b.streak = b.fast_role_streak[size_t(best_receiver)];
                }

                uint64_t donor_budget = 0;
                Counts donor_weights{};
                for (size_t c = 0; c < classes(); ++c) {
                    donor_weights[c] = donor_limits[c];
                    donor_budget = donor_limits[c] > UINT64_MAX - donor_budget
                                       ? UINT64_MAX
                                       : donor_budget + donor_limits[c];
                }

                // Do not wind up target debt.  Every update below is a
                // conservation-preserving donor/receiver pair, and no new
                // update is admitted until the public actuator settles the
                // previous one.
                if (any_shortage && donor_budget && b.actual == b.target) {
                    std::array<bool, kClasses> eligible_receivers{};
                    Counts receiver_weights{};
                    for (size_t c = 0; c < classes(); ++c) {
                        eligible_receivers[c] = receivers[c] && ready[c];
                        receiver_weights[c] = s.misses[c];
                    }
                    uint64_t receiver_count = 0;
                    for (size_t c = 0; c < classes(); ++c)
                        receiver_count += eligible_receivers[c];
                    if (receiver_count) {
                        // Scale the shared cap from the strongest ready miss
                        // fraction: 10% misses gives the one-percent floor,
                        // 50% gives five percent, and 100% reaches the ten
                        // percent ceiling.  Small pools remain one Region.
                        const uint64_t mild_cap =
                            std::max<uint64_t>(1, total / 100);
                        const uint64_t severity_cap = static_cast<uint64_t>(
                            static_cast<long double>(total) *
                            max_ready_miss_fraction / 10.0L);
                        const uint64_t cap = std::min<uint64_t>(
                            std::max<uint64_t>({1, mild_cap, severity_cap}),
                            std::max<uint64_t>(1, total / 10));
                        const uint64_t budget = std::min(cap, donor_budget);
                        Counts receiver_limits{};
                        receiver_limits.fill(UINT64_MAX);
                        const Counts gains = fast_split(receiver_weights,
                                                        eligible_receivers,
                                                        receiver_limits, budget);
                        uint64_t planned = 0;
                        for (size_t c = 0; c < classes(); ++c)
                            planned = gains[c] > UINT64_MAX - planned
                                          ? UINT64_MAX
                                          : planned + gains[c];
                        if (planned) {
                            const Counts losses = fast_split(
                                donor_weights, donors, donor_limits, planned);
                            uint64_t realized = 0;
                            for (size_t c = 0; c < classes(); ++c)
                                realized = losses[c] > UINT64_MAX - realized
                                               ? UINT64_MAX
                                               : realized + losses[c];
                            if (realized != planned)
                                throw std::logic_error(
                                    "fast Region budget split lost conservation");
                            if (planned) {
                                for (size_t c = 0; c < classes(); ++c) {
                                    if (gains[c]) b.target[c] += gains[c];
                                    if (losses[c]) b.target[c] -= losses[c];
                                    if (gains[c])
                                        b.fast_last_transfer_role[c] = 1;
                                    if (losses[c])
                                        b.fast_last_transfer_role[c] = 2;
                                }
                                s.transferred = planned;
                            }
                        }
                    }
                }
            } else {
                // Legacy controller: one receiver, one donor, one-percent
                // step, and the existing two-window hysteresis.
                // For pools below 100 Regions one Region is the indivisible step.
                const uint64_t step = std::max<uint64_t>(1, total / 100);
                int receiver = -1;
                int donor = -1;
                for (size_t r = 0; r < classes(); ++r) {
                    const bool shortage = s.attempts[r] >= 32 && s.misses[r] * 10 >= s.attempts[r];
                    if(!shortage) continue;
                    for(size_t d=0;d<classes();++d) {
                    if(d==r)continue;
                    const bool donor_quiet = s.attempts[d] == 0 || s.misses[d] * 20 < s.attempts[d];
                    const bool slack = s.free_bytes[d] / region_bytes_ >= step;
                    if (donor_quiet && slack && b.target[d] > 1 && total > 1 &&
                        (receiver<0 || s.misses[r]>s.misses[size_t(receiver)] ||
                         (r==size_t(receiver) && s.free_bytes[d]>s.free_bytes[size_t(donor)]))) {
                        receiver = int(r); donor=int(d);
                    }
                    }
                }
                if (receiver < 0) { b.direction = -1; b.streak = 0; }
                else if (receiver == b.direction) b.streak = std::min(2u, b.streak + 1);
                else { b.direction = receiver; b.streak = 1; }
                // Do not accumulate unfulfilled target changes: wait for the
                // background public-Region actuator to settle the previous debt.
                if (mode_ == Mode::Adaptive && b.streak >= 2 && b.actual == b.target) {
                    const size_t r = size_t(receiver), d = size_t(donor);
                    const uint64_t amount = std::min({step, b.target[d] - 1,
                                                      s.free_bytes[d] / region_bytes_});
                    b.target[r] += amount; b.target[d] -= amount;
                    s.transferred = amount;
                    b.streak = 0;
                }
            }
            s.actual = b.actual; s.target = b.target;
            s.applied = b.applied; s.streak = b.streak; b.last = s;
            if(six_enabled()) {
                if(total==0 && sum(s.attempts)==0)continue;
                for(size_t c=0;c<classes();++c)
                    std::cout << "simple_region_budget.six_window domain=" << name_ << " window=" << window << " bin=" << bin << " class=" << c
                              << " actual=" << s.actual[c] << " target=" << s.target[c] << " attempts=" << s.attempts[c] << " misses=" << s.misses[c]
                              << " public_free=" << s.free_bytes[c] << " transferred=" << s.transferred << " applied_total=" << s.applied << std::endl;
                continue;
            }
            if (total == 0 && s.attempts[0] + s.attempts[1] == 0) continue;
            std::cout << "simple_region_budget.window domain=" << name_
                      << " window=" << window << " bin=" << bin
                      << " mode=" << (mode_ == Mode::Adaptive ? "adaptive" : "fixed")
                      << " actual_cold=" << s.actual[0] << " actual_hot=" << s.actual[1]
                      << " target_cold=" << s.target[0] << " target_hot=" << s.target[1]
                      << " attempts_cold=" << s.attempts[0] << " attempts_hot=" << s.attempts[1]
                      << " misses_cold=" << s.misses[0] << " misses_hot=" << s.misses[1]
                      << " public_free_cold=" << s.free_bytes[0] << " public_free_hot=" << s.free_bytes[1]
                      << " transferred=" << s.transferred << " applied_total=" << s.applied
                      << " pending=" << (s.target[0] > s.actual[0] ? s.target[0] - s.actual[0] : s.actual[0] - s.target[0])
                      << std::endl;
        }
    }
};
inline Controller &local() { static Controller c("local", mode()); return c; }
inline Controller &remote() { static Controller c("remote", mode()); return c; }
} // namespace FarLib::simple_region_budget
