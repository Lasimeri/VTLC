#include "vtlc/codec.hpp"
#include <bit>
#include <cassert>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace vtlc;
int main() {
  std::mt19937 rng(1283);
  std::vector<Backend> backends{Backend::Scalar, Backend::AVX2,
                                Backend::AVX2_FMA, Backend::AVX512F};
  unsigned cases = 0;
  for (unsigned rep = 0; rep < 80; rep++) {
    uint64_t t = rep % 9, n = rep % 31;
    std::vector<uint64_t> shape{t, 3, n};
    std::vector<float> x(t * 3 * n);
    for (size_t i = 0; i < x.size(); i++) {
      uint32_t v = rng();
      if (rep % 4 == 0)
        v = i % (3 * n) ? 0x7f800001 : 0x80000000;
      if (rep % 4 == 1 && i >= 3 * n)
        std::memcpy(&v, &x[i - 3 * n], 4);
      std::memcpy(&x[i], &v, 4);
    }
    for (auto axis : {0u, 1u, 2u}) {
      Options o;
      o.time_axis = axis;
      o.tile_elements = 17;
      o.keyframe_interval = 3;
      auto encoded = compress(x, shape, o);
      for (auto be : backends)
        if (available(be)) {
          auto y = decompress(encoded, be);
          assert(y.bits.size() == x.size());
          assert(x.empty() ||
                 !std::memcmp(x.data(), y.bits.data(), x.size() * 4));
          o.backend = be;
          auto e = compress(x, shape, o);
          auto z = decompress(e, Backend::Scalar);
          assert(x.empty() ||
                 !std::memcmp(x.data(), z.bits.data(), x.size() * 4));
          cases++;
        }
    }
  }
  std::vector<uint64_t> shape{3, 19};
  std::vector<float> x(57);
  for (auto &f : x) {
    uint32_t b = rng();
    std::memcpy(&f, &b, 4);
  }
  auto good = compress(x, shape);
  for (size_t i = 0; i < good.size(); i++) {
    try {
      decompress(std::span<const uint8_t>(good).first(i));
      assert(false);
    } catch (const std::exception &) {
    }
  }
  for (int i = 0; i < 2000; i++) {
    auto bad = good;
    for (int j = 0; j < 1 + i % 4; j++)
      bad[rng() % bad.size()] ^= uint8_t(1u << (rng() % 8));
    try {
      (void)decompress(bad, Backend::Scalar, 1 << 20);
    } catch (const std::exception &) {
    }
  }
  std::cout << "passed " << cases
            << " cross-backend round trips, truncation and 2000 mutations\n";
}
