#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <x86intrin.h>

namespace FarLib::read_owner_diag {
constexpr size_t Capacity = 65536;
struct Record {
    std::atomic<uint64_t> key{0};
    std::atomic<uint64_t> submit{0}, cqe{0}, local{0}, handled{0}, waiter{0};
    std::atomic<uint64_t> poster_fibre{0}, poller_fibre{0}, waiter_fibre{0};
    std::atomic<uint32_t> client{0}, endpoint{0}, depth{0};
};
inline std::array<Record, Capacity> records;
inline std::array<std::atomic<uint32_t>, 512> pending{};
inline std::atomic<uint64_t> collisions{0};
inline bool enabled() {
    static const bool on = std::getenv("FARLIB_READ_OWNER_DIAG") != nullptr;
    return on;
}
inline uint64_t hash(uint64_t v) {
    v ^= v >> 30; v *= UINT64_C(0xbf58476d1ce4e5b9);
    v ^= v >> 27; v *= UINT64_C(0x94d049bb133111eb);
    return v ^ (v >> 31);
}
inline Record *find(uint64_t key, bool create = false) {
    if (!enabled()) return nullptr;
    const auto h = hash(key);
    if ((h & 1023) != 0) return nullptr;
    for (size_t i=0;i<8;++i) {
        auto &r=records[((h >> 10)+i) % Capacity];
        auto old=r.key.load(std::memory_order_acquire);
        if (old==key) return &r;
        if (!old && create && r.key.compare_exchange_strong(old,key))
            return &r;
        if (!old && !create) return nullptr;
    }
    if (create) ++collisions;
    return nullptr;
}
inline size_t lane(size_t client, size_t endpoint) {
    const auto i=client*8+endpoint;
    if (i>=pending.size()) std::abort();
    return i;
}
inline Record *begin(uint64_t key, size_t client, size_t endpoint, uint64_t fibre) {
    if (!enabled()) return nullptr;
    const auto depth=pending[lane(client,endpoint)].fetch_add(1)+1;
    auto *r=find(key,true);
    if (r && !r->submit.load()) {
        r->client=client; r->endpoint=endpoint; r->depth=depth;
        r->poster_fibre=fibre;
    }
    return r;
}
inline void accepted(Record *r) { if(r) r->submit=__rdtsc(); }
inline void received(uint64_t key, uint64_t fibre) {
    if(auto *r=find(key)) { r->cqe=__rdtsc(); r->poller_fibre=fibre; }
}
inline void completed(size_t client, size_t endpoint) {
    if(enabled()) pending[lane(client,endpoint)].fetch_sub(1);
}
inline void published(uint64_t key) { if(auto *r=find(key)) r->local=__rdtsc(); }
inline void finish(uint64_t key) { if(auto *r=find(key)) r->handled=__rdtsc(); }
inline void waited(Record *r, uint64_t fibre) {
    if(r) { r->waiter=__rdtsc(); r->waiter_fibre=fibre; }
}
inline void report() {
    if(!enabled()) return;
    for(auto &r:records) if(r.key.load() && r.submit.load())
        std::printf("read_owner key=%llu submit=%llu cqe=%llu local=%llu handled=%llu waiter=%llu poster=%llu poller=%llu waiting=%llu client=%u endpoint=%u depth=%u\n",
            (unsigned long long)r.key.load(),(unsigned long long)r.submit.load(),
            (unsigned long long)r.cqe.load(),(unsigned long long)r.local.load(),
            (unsigned long long)r.handled.load(),(unsigned long long)r.waiter.load(),
            (unsigned long long)r.poster_fibre.load(),(unsigned long long)r.poller_fibre.load(),
            (unsigned long long)r.waiter_fibre.load(),r.client.load(),r.endpoint.load(),r.depth.load());
    std::printf("read_owner collisions=%llu\n",(unsigned long long)collisions.load());
}
}
