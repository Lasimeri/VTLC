#include "vtlc/codec.hpp"
#include <array>
#include <cassert>
#include <span>
int main() {
  alignas(64) std::array<uint32_t, 272> x{}, a{}, b{}, ref{}, out{};
  for (size_t i = 0; i < x.size(); i++) {
    x[i] = uint32_t(i * 0x9e3779b9u);
    a[i] = uint32_t(i * 13);
    b[i] = uint32_t(i * 7);
  }
  for (size_t off : {0u, 1u})
    for (int p = 0; p <= 3; p++) {
      auto sx = std::span<const uint32_t>(x).subspan(off, 257);
      auto sa = std::span<const uint32_t>(a).subspan(off, 257);
      auto sb = std::span<const uint32_t>(b).subspan(off, 257);
      auto sr = std::span<uint32_t>(ref).subspan(off, 257);
      auto so = std::span<uint32_t>(out).subspan(off, 257);
      vtlc::predict_xor_words(sx, sa, sb, sr, p, vtlc::Backend::Scalar);
      for (auto be : {vtlc::Backend::AVX2, vtlc::Backend::AVX2_FMA,
                      vtlc::Backend::AVX512F})
        if (vtlc::available(be)) {
          vtlc::predict_xor_words(sx, sa, sb, so, p, be);
          for (size_t i = 0; i < 257; i++)
            assert(sr[i] == so[i]);
        }
    }
}
