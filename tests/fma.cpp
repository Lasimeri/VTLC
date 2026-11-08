#include "vtlc/codec.hpp"
#include <cassert>
#include <cstring>
#include <random>
#if defined(__x86_64__)
#include <immintrin.h>
#endif
int main() {
  constexpr size_t n = 8195;
  std::mt19937 rng(981);
  std::vector<float> x(n * 4);
  size_t index = 0;
  for (auto &f : x) {
    uint32_t b = rng();
    static constexpr uint32_t edge[]{
        0,          0x80000000, 1,          0x007fffff, 0x00800000, 0x7f7fffff,
        0xff7fffff, 0x7f800000, 0xff800000, 0x7f800001, 0x7fc01234, 0xff800001};
    if (index++ % 7 == 0)
      b = edge[rng() % (sizeof(edge) / sizeof(edge[0]))];
    std::memcpy(&f, &b, 4);
  }
  std::vector<uint64_t> shape{4, n};
  vtlc::Options o;
  o.predictor = 3;
  o.keyframe_interval = 4;
  o.tile_elements = 257;
  o.backend = vtlc::Backend::Scalar;
  auto reference = vtlc::compress(x, shape, o);
#if defined(__x86_64__)
  unsigned saved = _mm_getcsr();
#endif
  for (unsigned mode = 0; mode < 4; ++mode) {
#if defined(__x86_64__)
    unsigned unusual = ((saved & ~0x6000u) | (mode << 13) | 0x8040u | 0x1f80u);
    _mm_setcsr(unusual);
#endif
    for (auto backend : {vtlc::Backend::Scalar, vtlc::Backend::AVX2,
                         vtlc::Backend::AVX2_FMA, vtlc::Backend::AVX512F})
      if (vtlc::available(backend)) {
        o.backend = backend;
        auto encoded = vtlc::compress(x, shape, o);
        assert(encoded == reference);
        auto decoded = vtlc::decompress(encoded, backend);
        assert(!std::memcmp(x.data(), decoded.bits.data(), x.size() * 4));
#if defined(__x86_64__)
        assert(_mm_getcsr() == unusual);
#endif
      }
  }
#if defined(__x86_64__)
  _mm_setcsr(saved);
#endif
}
