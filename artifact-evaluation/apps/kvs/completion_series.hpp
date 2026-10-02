#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <sys/syscall.h>
#include <unistd.h>

namespace kvs_observation {
inline uint64_t monotonic_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline bool completion_series_enabled(const char *value) {
    if (!value || std::string(value) == "0") return false;
    if (std::string(value) == "1") return true;
    throw std::invalid_argument("FARLIB_KVS_COMPLETION_SERIES must be 0 or 1");
}

// Each fibre is the sole writer of its cache line. No shared increment or
// per-operation clock read is added; only the enabled template publishes.
class CompletionSeries {
public:
    static constexpr uint64_t interval_ns = 100000000;
    struct Row {
        uint64_t window_start_ns, window_end_ns, nominal_end_ns;
        uint64_t scan_begin_ns, scan_end_ns, completed, cumulative;
        bool final;
    };
private:
    struct alignas(64) Counter { std::atomic<uint64_t> completed{0}; };
    size_t count_;
    int cpu_;
    std::unique_ptr<Counter[]> counters_;
    std::thread observer_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool ready_ = false, started_ = false, stopped_ = false;
    int pin_error_ = 0;
    long observer_tid_ = 0;
    uint64_t start_ns_ = 0, end_ns_ = 0;
    std::vector<Row> rows_;

    uint64_t total() const {
        uint64_t n=0;
        for (size_t i=0;i<count_;++i)
            n += counters_[i].completed.load(std::memory_order_relaxed);
        return n;
    }
    void collect() {
        observer_tid_=syscall(SYS_gettid);
        if (cpu_ >= 0) {
            cpu_set_t mask;
            CPU_ZERO(&mask); CPU_SET(cpu_, &mask);
            pin_error_ = pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
        }
        std::unique_lock<std::mutex> lock(mutex_);
        ready_ = true;
        cv_.notify_all();
        cv_.wait(lock,[&]{return started_ || stopped_;});
        if (stopped_) return;
        uint64_t nominal = start_ns_ + interval_ns;
        uint64_t prior_time = start_ns_, prior_total = 0;
        while (!stopped_) {
            const auto deadline = std::chrono::steady_clock::time_point(
                std::chrono::nanoseconds(nominal));
            if (cv_.wait_until(lock,deadline,[&]{return stopped_;})) break;
            lock.unlock();
            const uint64_t scan_begin = monotonic_ns();
            const uint64_t cumulative = total();
            const uint64_t scan_end = monotonic_ns();
            rows_.push_back({prior_time,scan_end,nominal,scan_begin,scan_end,
                             cumulative-prior_total,cumulative,false});
            prior_time=scan_end; prior_total=cumulative;
            // If the OS delayed the observer, record the long actual window.
            // Never synthesize counts for time windows that were not sampled.
            nominal = start_ns_ +
                ((scan_end-start_ns_)/interval_ns+1)*interval_ns;
            lock.lock();
        }
    }
public:
    explicit CompletionSeries(size_t fibres, int cpu)
        : count_(fibres),cpu_(cpu),counters_(new Counter[fibres]) {
        if (fibres==0 || cpu>=CPU_SETSIZE || cpu < -1)
            throw std::invalid_argument("invalid completion observer setup");
        rows_.reserve(36002); // one hour at 100ms plus final boundary
        observer_=std::thread([this]{collect();});
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock,[&]{return ready_;});
        if (pin_error_) {
            stopped_=true; cv_.notify_all(); lock.unlock(); observer_.join();
            throw std::runtime_error("completion observer CPU pin failed");
        }
    }
    ~CompletionSeries() {
        if (observer_.joinable()) {
            { std::lock_guard<std::mutex> lock(mutex_); stopped_=true; }
            cv_.notify_all(); observer_.join();
        }
    }
    void start(uint64_t ns) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (started_ || stopped_) throw std::logic_error("series start repeated");
        start_ns_=ns; started_=true; cv_.notify_all();
    }
    void publish(size_t fibre, uint64_t cumulative) {
        counters_[fibre].completed.store(cumulative,std::memory_order_relaxed);
    }
    void finish(uint64_t ns, uint64_t expected) {
        { std::lock_guard<std::mutex> lock(mutex_); stopped_=true; }
        cv_.notify_all(); observer_.join();
        if (!started_ || ns<=start_ns_ || total()!=expected)
            throw std::runtime_error("completion series final receipt mismatch");
        end_ns_=ns;
        // The last worker may finish while a periodic snapshot is underway.
        // Discard snapshots crossing the request end, then close exactly once
        // using the final joined-worker count at that request boundary.
        while (!rows_.empty() && rows_.back().scan_end_ns>=ns) rows_.pop_back();
        const uint64_t prior_time=rows_.empty()?start_ns_:rows_.back().window_end_ns;
        const uint64_t prior_total=rows_.empty()?0:rows_.back().cumulative;
        if (prior_total>expected) throw std::runtime_error("series count reversed");
        rows_.push_back({prior_time,ns,ns,ns,ns,expected-prior_total,expected,true});
    }
    const std::vector<Row>& rows() const { return rows_; }
    void write(const std::string& prefix) const {
        std::ofstream csv(prefix+".csv");
        csv << "window_start_ns,window_end_ns,nominal_end_ns,scan_begin_ns,scan_end_ns,completed,cumulative_completed,final\n";
        for (const auto&r:rows_)
            csv<<r.window_start_ns<<','<<r.window_end_ns<<','<<r.nominal_end_ns<<','
               <<r.scan_begin_ns<<','<<r.scan_end_ns<<','<<r.completed<<','
               <<r.cumulative<<','<<r.final<<'\n';
        csv.close();
        if (!csv) throw std::runtime_error("completion CSV write failed");
        std::ofstream meta(prefix+".json");
        meta << "{\"enabled\":true,\"interval_ns\":"<<interval_ns
             <<",\"request_start_ns\":"<<start_ns_<<",\"request_end_ns\":"<<end_ns_
             <<",\"fibres\":"<<count_<<",\"collector_cpu\":"<<cpu_
             <<",\"collector_tid\":"<<observer_tid_
             <<",\"rows\":"<<rows_.size()<<",\"completed\":"<<rows_.back().cumulative
             <<",\"metric\":\"all completed GET+PUT+REMOVE; per-fibre cumulative snapshots\""
             <<",\"timing\":\"100ms target; use recorded actual window widths; snapshot scan bounds expose skew\"}\n";
        meta.close();
        if (!meta) throw std::runtime_error("completion metadata write failed");
    }
};
}
