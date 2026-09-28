#pragma once

#include <cstdint>

namespace FarLib::runtime_metadata {

// Retained compute-side metadata storage, not data/parity payload or RSS.
// Inline descriptors use sizeof; owned arrays/containers use their allocated
// capacities. Category bytes exclude measurement-only state, reported apart.
struct Snapshot {
    uint64_t region_bytes = 0;
    uint64_t group_bytes = 0;
    uint64_t stripe_bytes = 0;
    uint64_t mapping_bytes = 0;
    uint64_t span_bytes = 0;
    uint64_t measurement_aux_bytes = 0;
    uint64_t local_regions = 0;
    uint64_t remote_regions = 0;
    uint64_t stripes = 0;
    uint64_t group_slots = 0;
    uint64_t spans = 0;

    uint64_t metadata_bytes() const {
        return region_bytes + group_bytes + stripe_bytes + mapping_bytes +
               span_bytes;
    }

    void add(const Snapshot &other) {
        region_bytes += other.region_bytes;
        group_bytes += other.group_bytes;
        stripe_bytes += other.stripe_bytes;
        mapping_bytes += other.mapping_bytes;
        span_bytes += other.span_bytes;
        measurement_aux_bytes += other.measurement_aux_bytes;
        local_regions += other.local_regions;
        remote_regions += other.remote_regions;
        stripes += other.stripes;
        group_slots += other.group_slots;
        spans += other.spans;
    }
};

}  // namespace FarLib::runtime_metadata
