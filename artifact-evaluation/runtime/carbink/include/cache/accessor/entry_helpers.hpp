#pragma once
#include "cache/accessor.hpp"

namespace FarLib::cache {

inline bool check_fetch(FarObjectEntry *entry, fetch_ddl_t &ddl) {
    return Cache::get_default()->check_fetch(entry, ddl);
}

inline size_t check_cq() { return Cache::get_default()->check_cq(); }

inline bool at_local(far_obj_t obj) {
    return Cache::get_default()->at_local(obj);
}

}  // namespace FarLib::cache
