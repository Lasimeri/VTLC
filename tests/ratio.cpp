#include "vtlc/codec.hpp"
#include <cassert>
#include <cstring>
#include <random>
int main() {
  std::mt19937 rng(1337);
  for (int n = 1; n < 160; n++) {
    std::vector<float> x(4 * n);
    for (int i = 0; i < 4 * n; i++) {
      uint32_t v;
      switch (n % 4) {
      case 0:
        v = (rng() % 8 == 0) ? rng() : 0;
        break;
      case 1:
        v = (rng() % 1024) << 10;
        break;
      case 2:
        v = rng();
        break;
      default:
        v = 0x3f800000u + (rng() % 100);
        break;
      }
      std::memcpy(&x[i], &v, 4);
    }
    std::vector<uint64_t> s{4, uint64_t(n)};
    vtlc::Options o;
    o.tile_elements = 37;
    for (auto preset : {vtlc::Preset::Fast, vtlc::Preset::Balanced,
                        vtlc::Preset::Adaptive, vtlc::Preset::Ratio})
      for (int p = -1; p < 4; p++) {
        o.preset = preset;
        o.predictor = p;
        auto c = vtlc::compress(x, s, o);
        auto y = vtlc::decompress(c);
        assert(!std::memcmp(x.data(), y.bits.data(), x.size() * 4));
      }
  }
}
