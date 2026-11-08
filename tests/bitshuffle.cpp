#include "vtlc/codec.hpp"
#include <cassert>
#include <random>
#include <vector>
int main() {
  std::mt19937 rng(4);
  for (size_t n = 0; n < 300; n++) {
    std::vector<uint32_t> x(n);
    for (auto &v : x)
      v = rng();
    auto scalar = vtlc::bitshuffle_words(x, vtlc::Backend::Scalar);
    assert(vtlc::bitunshuffle_words(scalar, n, vtlc::Backend::Scalar) == x);
    for (auto be :
         {vtlc::Backend::AVX2, vtlc::Backend::AVX2_FMA, vtlc::Backend::AVX512F})
      if (vtlc::available(be)) {
        auto encoded = vtlc::bitshuffle_words(x, be);
        assert(encoded == scalar);
        assert(vtlc::bitunshuffle_words(encoded, n, be) == x);
      }
  }
}
