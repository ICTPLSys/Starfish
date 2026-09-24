#pragma once
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <x86intrin.h>
#include "utils/wait_trace.hpp"

namespace scope_diag {
inline const bool enabled=[] { const char *s=std::getenv("SCOPE_DIAG"); return s && *s=='1'; }();
inline const bool light_enabled=[] {
  const char *s=std::getenv("FARLIB_SCOPE_DIAG_LIGHT");
  return s && s[0]=='1' && s[1]=='\0';
}();
inline const bool scope_wait_enabled=[] {
  const char *s=std::getenv("FARLIB_SCOPE_WAIT_DIAG");
  return s && s[0]=='1' && s[1]=='\0';
}();
enum Phase : uint64_t { UNKNOWN, COMPUTE, ACCESS, ALLOC, REFILL, RDMA_POST,
                       RDMA_WAIT, CQ_POLL, YIELD, MUTEX, COND, WC_HANDLE,
                       REMOTE_FREE };
inline constexpr uint64_t OUT=~uint64_t(0);
struct alignas(256) Slot {
  std::atomic<uint64_t> fid{0}, seq{0}, generation{OUT}, phase{UNKNOWN};
  std::atomic<uint64_t> phase_start{0}, op_start{0}, ops{0}, allocs{0}, refills{0};
  std::atomic<uint64_t> last_ack{0}, pending{0}, posts{0}, polls{0}, nonempty{0};
  std::atomic<uint64_t> yield_start{0}, yield_end{0}, poll_start{0}, poll_end{0};
};
inline Slot slots[128];
inline std::atomic<uintptr_t> keys[1024]{};
inline std::atomic<Slot*> values[1024]{};
inline uint64_t (*inspect_entry)(uintptr_t)=nullptr;
inline std::atomic<uint64_t> flip_epoch{0}, flip_old{OUT};
inline uint64_t last_sample=0;
inline uint64_t sample_seq=0;
inline uint64_t scope_wait_last_count=OUT;
inline size_t hash(uintptr_t x) { return ((x>>6)*11400714819323198485ull)>>54; }
inline Slot *current(void *f) {
  if (!enabled && !light_enabled && !scope_wait_enabled) return nullptr;
  uintptr_t id=reinterpret_cast<uintptr_t>(f);
  size_t h=hash(id);
  for (size_t n=0;n<1024;++n,h=(h+1)&1023) {
    auto k=keys[h].load(std::memory_order_acquire);
    if (k==id) {
      auto *slot=values[h].load(std::memory_order_acquire);
      // Fibre slots are reused by tid in the 24/48/72 sequence while old hash
      // keys remain.  Reject a stale key if its slot now belongs to a new fid.
      if (slot && slot->fid.load(std::memory_order_acquire)==id) return slot;
      return nullptr;
    }
    if (!k) return nullptr;
  }
  return nullptr;
}
struct Write {
  Slot *s;
  explicit Write(Slot *p):s(p) { s->seq.fetch_add(1,std::memory_order_acq_rel); }
  ~Write() { s->seq.fetch_add(1,std::memory_order_release); }
};
inline size_t index(Slot *s) { return static_cast<size_t>(s-slots); }
inline Slot *register_fibre(size_t tid,void *f) {
  if ((!enabled && !light_enabled && !scope_wait_enabled) || tid>=128) return nullptr;
  Slot *s=&slots[tid];
  uintptr_t id=reinterpret_cast<uintptr_t>(f);
  s->fid.store(id,std::memory_order_release);
  size_t h=hash(id);
  for (size_t n=0;n<1024;++n,h=(h+1)&1023) {
    uintptr_t empty=0;
    if (keys[h].compare_exchange_strong(empty,id) || empty==id) {
      values[h].store(s,std::memory_order_release);
      wait_trace::emit("sd_register",tid,0,0,id);
      return s;
    }
  }
  std::abort();
}
struct Guard {
  Slot *s; uint64_t old, old_start; Phase p; bool light_only;
  Guard(void *f,Phase value):s(current(f)),old(0),old_start(0),p(value),
      light_only((light_enabled || scope_wait_enabled) && !enabled) {
    if (!s) return;
    if (light_only) {
      old=s->phase.load(std::memory_order_relaxed);
      s->phase.store(p,std::memory_order_relaxed);
      return;
    }
    Write w(s);
    old=s->phase.load(std::memory_order_relaxed);
    old_start=s->phase_start.load(std::memory_order_relaxed);
    auto now=__rdtsc();
    s->phase_start.store(now,std::memory_order_relaxed);
    s->phase.store(p,std::memory_order_relaxed);
    if (p==YIELD) s->yield_start.store(now,std::memory_order_relaxed);
    if (p==CQ_POLL) s->poll_start.store(now,std::memory_order_relaxed);
    if (p==ALLOC) s->allocs.store(s->allocs.load()+1,std::memory_order_relaxed);
    if (p==REFILL) s->refills.store(s->refills.load()+1,std::memory_order_relaxed);
  }
  ~Guard() {
    if (!s) return;
    if (light_only) {
      s->phase.store(old,std::memory_order_relaxed);
      return;
    }
    Write w(s);auto now=__rdtsc();
    if (p==YIELD) s->yield_end.store(now,std::memory_order_relaxed);
    if (p==CQ_POLL) s->poll_end.store(now,std::memory_order_relaxed);
    s->phase_start.store(old_start,std::memory_order_relaxed);
    s->phase.store(old,std::memory_order_relaxed);
  }
};
inline void begin_op(Slot *s) {
  if (!s) return;
  if ((light_enabled || scope_wait_enabled) && !enabled) {
    s->phase.store(COMPUTE,std::memory_order_relaxed);
    return;
  }
  Write w(s);
  auto now=__rdtsc(); s->op_start.store(now,std::memory_order_relaxed);
  s->phase_start.store(now,std::memory_order_relaxed);s->phase.store(COMPUTE,std::memory_order_relaxed);
}
inline void end_op(Slot *s) {
  if (!s) return;
  if ((light_enabled || scope_wait_enabled) && !enabled) {
    s->pending.store(0,std::memory_order_relaxed);
    s->phase.store(COMPUTE,std::memory_order_relaxed);
    return;
  }
  Write w(s);
  s->ops.store(s->ops.load()+1,std::memory_order_relaxed);
  s->pending.store(0,std::memory_order_relaxed);
  s->phase.store(COMPUTE,std::memory_order_relaxed);s->phase_start.store(__rdtsc(),std::memory_order_relaxed);
}
inline void set_pending(void *f,uintptr_t entry) {
  auto *s=current(f);if(!s)return;
  if((light_enabled || scope_wait_enabled) && !enabled){s->pending.store(entry,std::memory_order_relaxed);return;}
  Write w(s);s->pending.store(entry,std::memory_order_relaxed);
}
inline void clear_pending(void *f) { set_pending(f,0); }
inline void posted(void *f) {
  if(!enabled)return;auto *s=current(f);if(!s)return;Write w(s);s->posts.store(s->posts.load()+1,std::memory_order_relaxed);
}
inline void refill(void *f) {
  if(!enabled && !scope_wait_enabled)return;auto *s=current(f);if(!s)return;
  if(scope_wait_enabled) {
    uint64_t count;
    if(enabled) {
      Write w(s);
      count=s->refills.load(std::memory_order_relaxed)+1;
      s->refills.store(count,std::memory_order_relaxed);
    } else {
      count=s->refills.fetch_add(1,std::memory_order_relaxed)+1;
    }
    wait_trace::emit("scope_refill",index(s),count,
                     s->phase.load(std::memory_order_relaxed),
                     reinterpret_cast<uintptr_t>(f));
    return;
  }
  Write w(s);s->refills.store(s->refills.load()+1,std::memory_order_relaxed);
}
inline void cq_result(void *f,size_t n) {
  if(!enabled)return;auto *s=current(f);if(!s)return;Write w(s);
  s->polls.store(s->polls.load()+1,std::memory_order_relaxed);
  if(n)s->nonempty.store(s->nonempty.load()+1,std::memory_order_relaxed);
}
inline void on_enter(void *f,uint64_t state) {
  if(!enabled && !scope_wait_enabled)return;auto *s=current(f);if(!s)return;
  if(scope_wait_enabled) {
    s->generation.store(state,std::memory_order_relaxed);
    wait_trace::emit("scope_enter",index(s),state,
                     s->refills.load(std::memory_order_relaxed),
                     reinterpret_cast<uintptr_t>(f));
    if(!enabled)return;
  }
  {Write w(s);s->generation.store(state,std::memory_order_relaxed);s->last_ack.store(__rdtsc(),std::memory_order_relaxed);}
  wait_trace::emit("sd_enter",index(s),state,0,reinterpret_cast<uintptr_t>(f));
}
inline void before_update(void *f,uint64_t old,uint64_t next) {
  if(!enabled && !scope_wait_enabled)return;auto *s=current(f);if(!s)return;
  if(scope_wait_enabled) {
    wait_trace::emit("scope_update_before",index(s),old,next,
                     reinterpret_cast<uintptr_t>(f));
    const uint64_t phase=s->phase.load(std::memory_order_relaxed);
    const uint64_t pending=s->pending.load(std::memory_order_relaxed);
    wait_trace::emit("scope_update_before_state",index(s),
                     s->refills.load(std::memory_order_relaxed),
                     (phase<<56)|(pending&0x00ffffffffffffffULL),
                     reinterpret_cast<uintptr_t>(f));
    if(!enabled)return;
  }
  wait_trace::emit("sd_ack_begin",index(s),old,next,reinterpret_cast<uintptr_t>(f));
}
inline void after_update(void *f,uint64_t old,uint64_t next) {
  if(!enabled && !scope_wait_enabled)return;auto *s=current(f);if(!s)return;
  if(scope_wait_enabled) {
    s->generation.store(next,std::memory_order_relaxed);
    s->last_ack.store(__rdtsc(),std::memory_order_relaxed);
    wait_trace::emit("scope_update_after",index(s),old,next,
                     reinterpret_cast<uintptr_t>(f));
    const uint64_t phase=s->phase.load(std::memory_order_relaxed);
    const uint64_t pending=s->pending.load(std::memory_order_relaxed);
    wait_trace::emit("scope_update_after_state",index(s),
                     s->refills.load(std::memory_order_relaxed),
                     (phase<<56)|(pending&0x00ffffffffffffffULL),
                     reinterpret_cast<uintptr_t>(f));
    if(!enabled)return;
  }
  {Write w(s);s->generation.store(next,std::memory_order_relaxed);s->last_ack.store(__rdtsc(),std::memory_order_relaxed);}
  wait_trace::emit("sd_ack_end",index(s),old,next,reinterpret_cast<uintptr_t>(f));
}
inline void on_exit(void *f,uint64_t old) {
  if(!enabled && !scope_wait_enabled)return;auto *s=current(f);if(!s)return;
  if(scope_wait_enabled) {
    s->generation.store(OUT,std::memory_order_relaxed);
    s->last_ack.store(__rdtsc(),std::memory_order_relaxed);
    wait_trace::emit("scope_exit",index(s),old,
                     s->refills.load(std::memory_order_relaxed),
                     reinterpret_cast<uintptr_t>(f));
    if(!enabled)return;
  }
  {Write w(s);s->generation.store(OUT,std::memory_order_relaxed);s->last_ack.store(__rdtsc(),std::memory_order_relaxed);}
  wait_trace::emit("sd_exit",index(s),old,OUT,reinterpret_cast<uintptr_t>(f));
}
inline void begin_flip(uint64_t epoch,uint64_t old) {
  if(!enabled && !scope_wait_enabled)return;
  flip_old.store(old);flip_epoch.store(epoch);last_sample=0;scope_wait_last_count=OUT;
  if(scope_wait_enabled) {
    wait_trace::emit("scope_flip_begin",old,epoch,0);
    if(!enabled)return;
  }
  wait_trace::emit("sd_flip_begin",old,epoch,0);
}
inline void snapshot(uint64_t epoch,uint64_t old_count) {
  if(scope_wait_enabled &&
     (scope_wait_last_count==OUT || old_count==0)) {
    wait_trace::emit("scope_old_count",flip_old.load(),epoch,old_count);
    scope_wait_last_count=old_count;
  }
  if(!enabled)return;
  auto now=__rdtsc();
  // Nominal 100us at ~2.8GHz. Actual sample TSC is retained; no fixed-spacing assumption.
  if(last_sample && now-last_sample<280000)return;
  last_sample=now;const auto sample=++sample_seq;
  wait_trace::emit("sd_scan",sample,epoch,old_count);
  const auto old=flip_old.load();
  uint64_t accepted=0,unstable=0;
  for(size_t i=0;i<128;++i) {
    auto &s=slots[i];auto fid=s.fid.load(std::memory_order_acquire);if(!fid)continue;
    auto seq=s.seq.load(std::memory_order_acquire);if(seq&1){++unstable;continue;}
    auto gen=s.generation.load(std::memory_order_relaxed);if(gen!=old)continue;
    auto phase=s.phase.load(),phase_start=s.phase_start.load(),op_start=s.op_start.load();
    auto ops=s.ops.load(),refills=s.refills.load(),allocs=s.allocs.load(),ack=s.last_ack.load();
    auto pending=s.pending.load(),posts=s.posts.load(),polls=s.polls.load(),nonempty=s.nonempty.load();
    auto ys=s.yield_start.load(),ye=s.yield_end.load(),ps=s.poll_start.load(),pe=s.poll_end.load();
    if(seq!=s.seq.load(std::memory_order_acquire)){++unstable;continue;}
    uint64_t entry_state=pending && inspect_entry ? inspect_entry(pending) : OUT;
    ++accepted;
    wait_trace::emit("sd_sample",i,epoch,phase,fid);
    wait_trace::emit("sd_progress",i,ops,refills,fid);
    wait_trace::emit("sd_detail",i,phase_start,op_start,fid);
    wait_trace::emit("sd_io",i,polls,nonempty,fid);
    wait_trace::emit("sd_pending",i,pending,entry_state,fid);
    wait_trace::emit("sd_ack_state",i,ack,gen,fid);
    wait_trace::emit("sd_yield",i,ys,ye,fid);
    wait_trace::emit("sd_polltime",i,ps,pe,fid);
    wait_trace::emit("sd_alloc",i,allocs,posts,fid);
  }
  wait_trace::emit("sd_scan_end",sample,accepted,unstable);
}
inline void end_flip(uint64_t epoch) {
  if(!enabled && !scope_wait_enabled)return;
  if(scope_wait_enabled)wait_trace::emit("scope_flip_end",flip_old.load(),epoch,0);
  if(enabled)wait_trace::emit("sd_flip_end",flip_old.load(),epoch,0);
  flip_epoch.store(0);
}
}
