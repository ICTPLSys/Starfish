#include "cache/entry.hpp"

#include <cassert>
#include <cstdint>
#include <memory>

namespace {

struct Context {
    explicit Context(int value) : value(value) {}
    ~Context() { ++destroyed; }

    int value;
    static int destroyed;
};

int Context::destroyed = 0;

bool add(void *destination, size_t bytes, const void *opaque, uint64_t arg) {
    if (bytes != sizeof(int) || opaque == nullptr) return false;
    *static_cast<int *>(destination) =
        static_cast<const Context *>(opaque)->value + static_cast<int>(arg);
    return true;
}

bool subtract(void *destination, size_t bytes, const void *opaque,
              uint64_t arg) {
    if (bytes != sizeof(int) || opaque == nullptr) return false;
    *static_cast<int *>(destination) =
        static_cast<const Context *>(opaque)->value - static_cast<int>(arg);
    return true;
}

}  // namespace

int main() {
    using FarLib::cache::FarObjectEntry;
    using FarLib::cache::LOCAL;
    using FarLib::cache::MARKED;
    using FarLib::cache::REMOTE;
    using FarLib::recompute::Inputs;

static_assert(sizeof(FarObjectEntry) == 48);

    Context::destroyed = 0;
    auto context = std::make_shared<Context>(40);
    Inputs inputs = Inputs::immutable_compute_owned(context);
    FarObjectEntry source;
    source.reset<true>(reinterpret_cast<void *>(0x1000),
                       FarObjectEntry::RemoteAddrInvalid48, sizeof(int));
    assert(source.try_bind_recompute_recipe(inputs, add, 2));
    assert(source.has_recompute_recipe());
    assert(source.is_recomputable());

    int result = 0;
    assert(source.recompute_binding().inputs->invoke(
        &result, sizeof(result), source.recompute_binding().arg));
    assert(result == 42);

    // A second bind on the entry is rejected, and this input set keeps the
    // callback chosen by its first successful registration.
    assert(!source.try_bind_recompute_recipe(inputs, add, 3));
    FarObjectEntry conflict;
    conflict.reset<true>(reinterpret_cast<void *>(0x2000),
                         FarObjectEntry::RemoteAddrInvalid48, sizeof(int));
    assert(!conflict.try_bind_recompute_recipe(inputs, subtract, 2));

    // A remote entry is too late to bind.
    FarObjectEntry late;
    late.reset<true>(reinterpret_cast<void *>(0x3000), 0x4000, sizeof(int));
    assert(!late.try_bind_recompute_recipe(inputs, add, 2));

    // Move transfers the intrusive reference without an extra retain/release.
    FarObjectEntry moved;
    moved.set_free();
    moved.take_recompute_recipe_from(source);
    assert(source.recompute_binding().inputs == nullptr);
    assert(moved.has_recompute_recipe());
    context.reset();
    inputs.reset();
    assert(Context::destroyed == 0);
    result = 0;
    auto binding = moved.recompute_binding();
    assert(binding.inputs->invoke(&result, sizeof(result), binding.arg));
    assert(result == 42);

    moved.set_free();
    assert(!moved.has_recompute_recipe());
    assert(Context::destroyed == 1);

    // A flag-only annotation remains a flag-only annotation.
    FarObjectEntry marked;
    marked.reset<true>(reinterpret_cast<void *>(0x5000),
                       FarObjectEntry::RemoteAddrInvalid48, sizeof(int));
    auto state = marked.load_state();
    state.state = MARKED;
    marked.set_state(state);
    assert(marked.try_mark_recomputable_before_eviction());
    assert(marked.is_recomputable());
    assert(!marked.has_recompute_recipe());
    assert(marked.load_state().state == LOCAL);

    // No recipe can be created after a remote state transition.
    state = marked.load_state();
    state.state = REMOTE;
    marked.set_state(state);
    assert(!marked.try_bind_recompute_recipe(
        Inputs::immutable_compute_owned<Context>(std::make_shared<Context>(7)),
        add, 1));
}
