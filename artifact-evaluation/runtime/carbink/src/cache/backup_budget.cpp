#include "cache/accounting/batched_backup_budget.hpp"

#ifndef NO_REMOTE
#include "libfibre/fibre.h"

namespace FarLib::cache {

BatchedBackupBudget::Handle &current_backup_budget_handle() {
    static const size_t key = Fibre::key_create([](void *p) {
        delete static_cast<BatchedBackupBudget::Handle *>(p);
    });
    Fibre *fibre = fibre_self();
    auto *handle = static_cast<BatchedBackupBudget::Handle *>(fibre->getspecific(key));
    if (!handle) {
        handle = new BatchedBackupBudget::Handle();
        fibre->setspecific(key, handle);
    }
    return *handle;
}

BatchedBackupBudget::Handle &current_backup_thread_handle() {
    static thread_local BatchedBackupBudget::Handle handle;
    return handle;
}

}  // namespace FarLib::cache
#endif
