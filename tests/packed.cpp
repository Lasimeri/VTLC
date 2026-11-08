#include "vtlc/codec.hpp"
#include <cassert>
#include <cstring>
#include <random>
int main() {
  std::mt19937 g(7);
  for (unsigned width : {8u, 16u})
    for (unsigned n = 1; n < 260; n++) {
      std::vector<float> input(n);
      for (unsigned i = 0; i < n; i++) {
        uint32_t b = (g() & ((1u << width) - 1)) | (1u << (width - 1));
        std::memcpy(&input[i], &b, 4);
      }
      std::vector<uint64_t> shape{1, n};
      vtlc::Options o;
      o.preset = vtlc::Preset::Fast;
      o.predictor = 0;
      o.backend = vtlc::Backend::Scalar;
      auto ref = vtlc::compress(input, shape, o);
      for (auto be : {vtlc::Backend::Scalar, vtlc::Backend::AVX2,
                      vtlc::Backend::AVX2_FMA, vtlc::Backend::AVX512F})
        if (vtlc::available(be)) {
          o.backend = be;
          auto enc = vtlc::compress(input, shape, o);
          assert(enc == ref);
          auto dec = vtlc::decompress(enc, be);
          assert(!std::memcmp(input.data(), dec.bits.data(), n * 4));
        }
    }
}
