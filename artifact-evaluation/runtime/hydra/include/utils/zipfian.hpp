// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ALEX/blob/master/src/benchmark/zipf.h

#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>

template <bool sorted>
class ZipfianGenerator {
public:
    static constexpr double ZIPFIAN_CONSTANT = 0.99;
    static constexpr double ZETAN = 26.46902820178302;
    int num_keys_;
    double alpha_;
    double eta_;
    double zetan_;
    double zipfian_constant_;
    std::uniform_real_distribution<double> dis_;

    explicit ZipfianGenerator(int num_keys,
                              double zipfian_constant = ZIPFIAN_CONSTANT)
        : num_keys_(num_keys), dis_(0, 1), zipfian_constant_(zipfian_constant) {
        if (zipfian_constant_ == 0.0) {
            if (num_keys_ <= 0) {
                throw std::invalid_argument("uniform key domain must be positive");
            }
            // Uniform object IDs need no scrambling. FNV(rank) % N is not
            // a permutation: it merges ranks and makes some IDs unreachable.
            zetan_ = num_keys_;
            alpha_ = eta_ = 1.0;
            return;
        }
        zetan_ = zeta(num_keys);
        double zeta2theta = zeta(2);
        alpha_ = 1. / (1. - zipfian_constant);
        eta_ = (1 - std::pow(2. / num_keys_, 1 - zipfian_constant)) /
               (1 - zeta2theta / zetan_);
        // eta_ = (1 - std::pow(2. / num_keys_, 1 - zipfian_constant)) /
            //    (1 - zeta2theta / ZETAN);
    }

    template <typename G>
    int nextValue(G& gen) {
        if (zipfian_constant_ == 0.0) {
            // Integer distribution avoids modulo bias, including non-power-of-
            // two domains. Apply to both sorted and scrambled generator modes.
            return std::uniform_int_distribution<int>(0, num_keys_ - 1)(gen);
        }
        double u = dis_(gen);
        double uz = u * zetan_;
        // double uz = u * ZETAN;

        int ret;
        if (uz < 1.0) {
            ret = 0;
        } else if (uz < 1.0 + std::pow(0.5, zipfian_constant_)) {
            ret = 1;
        } else {
            ret = (int)(num_keys_ * std::pow(eta_ * u - eta_ + 1, alpha_));
        }

        if constexpr (!sorted) {
            ret = fnv1a(ret) % num_keys_;
        }

        return ret;
    }

    template <typename G>
    int operator()(G& g) {
        return nextValue(g);
    }

    double zeta(long n) {
        double sum = 0.0;
        for (long i = 0; i < n; i++) {
            sum += 1 / std::pow(i + 1, zipfian_constant_);
        }
        return sum;
    }

    // FNV hash from https://create.stephan-brumme.com/fnv-hash/
    static const uint32_t PRIME = 0x01000193;  //   16777619
    static const uint32_t SEED = 0x811C9DC5;   // 2166136261
    /// hash a single byte
    inline uint32_t fnv1a(unsigned char oneByte, uint32_t hash = SEED) {
        return (oneByte ^ hash) * PRIME;
    }
    /// hash a 32 bit integer (four bytes)
    inline uint32_t fnv1a(int fourBytes, uint32_t hash = SEED) {
        const unsigned char* ptr = (const unsigned char*)&fourBytes;
        hash = fnv1a(*ptr++, hash);
        hash = fnv1a(*ptr++, hash);
        hash = fnv1a(*ptr++, hash);
        return fnv1a(*ptr, hash);
    }
};
