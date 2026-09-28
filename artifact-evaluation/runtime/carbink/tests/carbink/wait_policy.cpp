#ifdef NDEBUG
#undef NDEBUG
#endif
#include "cache/carbink/wait_policy.hpp"
#include <cassert>
#include <iostream>
int main() {
    using FarLib::cache::carbink::poll_global_wait_cqs;
    // A stable, healthy Carbink page has an exact READ client/endpoint.
    assert(!poll_global_wait_cqs(true,true,false));
    // Any failure retains the established all-CQ recovery assistance.
    assert(poll_global_wait_cqs(true,true,true));
    // Generic/non-page paths retain their existing progress contract.
    assert(poll_global_wait_cqs(true,false,false));
    assert(poll_global_wait_cqs(true,false,true));
    assert(poll_global_wait_cqs(false,true,false));
    assert(poll_global_wait_cqs(false,true,true));
    assert(poll_global_wait_cqs(false,false,false));
    assert(poll_global_wait_cqs(false,false,true));
    using FarLib::cache::carbink::yield_periodically_in_wait;
    assert(!yield_periodically_in_wait(true,true,false));
    assert(yield_periodically_in_wait(true,true,true));
    assert(yield_periodically_in_wait(true,false,false));
    assert(yield_periodically_in_wait(true,false,true));
    assert(yield_periodically_in_wait(false,true,false));
    assert(yield_periodically_in_wait(false,true,true));
    assert(yield_periodically_in_wait(false,false,false));
    assert(yield_periodically_in_wait(false,false,true));
    std::cout<<"CARBINK_WAIT_POLICY_PASS cases=16\n";
}
