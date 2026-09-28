// CPU-only: policy, quota isolation, survivor placement and partial-group handoff.
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "cache/alloc/ec_rmw_fallback_policy.hpp"
#include "cache/alloc/ec_direct_group_builder.hpp"

namespace FarLib {
static rdma::Configure fixture;
const rdma::Configure &get_config() { return fixture; }
}
namespace FarLib::allocator::remote { RemoteGlobalHeap remote_global_heap; }

using namespace FarLib::cache;
using namespace FarLib::cache::ec_batch;

static bool host_allocate(void *, size_t bytes, EcDirectWriteBank::RegisteredBank *out) {
    void *p = std::aligned_alloc(4096, (bytes+4095)&~size_t(4095));
    if (!p) return false;
    std::memset(p,0,(bytes+4095)&~size_t(4095));
    out->base=p; out->bytes=bytes; out->lkey=1; out->registration=nullptr;
    return true;
}
static void host_release(void *, EcDirectWriteBank::RegisteredBank &bank) {
    std::free(bank.base); bank={};
}

int main() {
    using ec_rmw_runtime::allow_read_failure_fallback;
    using ec_rmw_runtime::allow_capacity_replacement;
    using ec_rmw_runtime::allow_background_bank_yield;
    for (unsigned background = 0; background < 2; ++background)
      for (unsigned dead = 0; dead < 2; ++dead)
        for (unsigned ready = 0; ready < 2; ++ready)
          assert(allow_background_bank_yield(background, dead, ready) ==
                 (background && dead && !ready));
    for (unsigned failed = 0; failed < 2; ++failed)
      for (unsigned eligible = 0; eligible < 2; ++eligible)
        for (unsigned background = 0; background < 2; ++background)
          assert(allow_capacity_replacement(failed, eligible, background) ==
                 (failed && (eligible || background)));
    assert(!FarLib::fixture.ft_rmw_read_failure_fallback);
    for (unsigned enabled=0;enabled<2;++enabled)
      for (unsigned sided=0;sided<2;++sided)
       for (unsigned errors=0;errors<8;++errors)
        for (unsigned terminal=0;terminal<8;++terminal)
         for (unsigned writes=0;writes<2;++writes)
          assert(allow_read_failure_fallback(enabled,sided,errors,terminal,writes) ==
                 (enabled && sided && errors && terminal==7 && !writes));
    setenv("FARLIB_EC_BENCHMARK_PHASED","1",1);
    auto &c=FarLib::fixture;
    c.server_count=7; c.server_buffer_size=4ull<<20; c.client_buffer_size=4ull<<20;
    c.remote_total=c.server_buffer_size*c.server_count;
    c.mapping_type=FarLib::rdma::Configure::MAPPING_RANGE;
    c.ft_method_type=FarLib::rdma::Configure::FT_EC_BATCH; c.ft_method="ec_batch";
    c.exclusive_cache=true; c.ft_incremental_update=true; c.ft_incremental_one_sided=true;
    c.ft_ec_data_shards=4; c.ft_ec_parity_shards=2; c.ft_small_object_cutoff=4096;
    c.ft_small_stripe_shard_size_bytes=FarLib::rdma::Configure::ft_ec_shard_size_bytes;
    FarLib::allocator::remote::remote_global_heap.register_remote(c.remote_total);
    SmallObjectStripeManager manager;
    manager.init(c.remote_total,c.ft_small_stripe_shard_size_bytes);
    assert(manager.enabled());
    // Populate one full pre-fault physical stripe, leaving no free healthy
    // group that could hide the zero healthy growth budget below.
    const size_t groups_per_stripe = c.ft_small_stripe_shard_size_bytes / 512;
    std::vector<SmallObjectStripeManager::SlotGroupHandle> old(groups_per_stripe);
    for (auto &g : old) assert(manager.allocate_slot_group(512, &g));
    const uint64_t quantum = 4 * c.ft_small_stripe_shard_size_bytes;
    EcDirectWriteBank bank(1); assert(bank.init(nullptr,host_allocate,host_release));
    EcDirectGroupBuilder builder(&manager,&bank,0,true);
    alignas(64) unsigned char object[512]; std::memset(object,0x5a,sizeof(object));
    uint64_t address=FarLib::allocator::remote::InvalidRemoteAddr;
    FarLib::ec_benchmark_phase::begin_work();
    assert(builder.add_object(object,sizeof(object),&address,0,true)==EcBatchStatus::kInvalidArgument);
    c.ft_rmw_read_failure_fallback=true;
    assert(!manager.recovery_replacement_enabled());
    assert(!allow_capacity_replacement(manager.recovery_replacement_enabled(), false, true));
    assert(builder.add_object(object,sizeof(object),&address,0,true)==EcBatchStatus::kManagerRejected);
    // Normal admission still uses the zero-sized healthy backup growth cap.
    assert(builder.add_object(object,sizeof(object),&address,0)==EcBatchStatus::kManagerRejected);
    assert(manager.mark_endpoint_dead(0));
    // Ordinary/no-old-address sources must make progress after a real failure,
    // but use exactly the same bounded quota tested below.
    assert(!manager.data_address_needs_replacement(FarLib::allocator::remote::InvalidRemoteAddr));
    assert(!allow_capacity_replacement(manager.recovery_replacement_enabled(), false, false));
    assert(allow_capacity_replacement(manager.recovery_replacement_enabled(), false, true));
    assert(manager.data_address_needs_replacement(old[0].segments[0].addr));
    assert(builder.add_object(object,sizeof(object),&address,0,true)==EcBatchStatus::kOk);
    assert(address!=FarLib::allocator::remote::InvalidRemoteAddr);
    assert(manager.backup_growth_data_bytes()==0);
    assert(manager.recovery_replacement_data_bytes()==quantum);
    assert(manager.recovery_replacement_data_limit_bytes()==quantum);
    assert(builder.flush()==EcBatchStatus::kOk);
    assert(!builder.group_open() && builder.pending_count()==1);
    auto token=builder.front_token();
    auto *entry=bank.get(token); assert(entry);
    auto id=entry->record.group.id;
    assert(entry->record.live_count==1 && entry->record.objects[0]==object);
    for (const auto &segment:entry->record.group.segments) assert(segment.endpoint_idx!=0);
    builder.commit_pending();
    for (unsigned i=0;i<5;++i) assert(bank.complete_segment(token,i,true)==nullptr);
    assert(!bank.release(token)); // Local source may not be released early.
    auto *done=bank.complete_segment(token,5,false);
    assert(done && done->recoverable());
    assert(bank.release(token));
    assert(manager.mark_dead_group(id));
    // Fill that one replacement stripe. Reusing its remaining groups costs
    // no extra credit; a second fresh stripe must fail, not bypass the cap.
    SmallObjectStripeManager::SlotGroupHandle exhausted;
    for (size_t i=0;i<groups_per_stripe;++i)
        assert(manager.allocate_slot_group(512,&exhausted,false,true));
    assert(!manager.allocate_slot_group(512,&exhausted,false,true));
    assert(!manager.mark_endpoint_dead(0));
    assert(!manager.allocate_slot_group(512,&exhausted,false,true));
    assert(manager.recovery_replacement_data_bytes()==quantum);
    assert(manager.recovery_replacement_data_limit_bytes()==quantum);
    assert(manager.backup_growth_data_bytes()==0);
    // Config-on must not manufacture six distinct live endpoints after two losses.
    manager.mark_endpoint_dead(1);
    address=FarLib::allocator::remote::InvalidRemoteAddr;
    assert(builder.add_object(object,sizeof(object),&address,0,true)==EcBatchStatus::kManagerRejected);
    assert(address==FarLib::allocator::remote::InvalidRemoteAddr);
    std::puts("RMW_READ_FALLBACK_CPU_PASS");
}
