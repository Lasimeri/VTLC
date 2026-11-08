#include "vtlc/codec.hpp"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
int main() {
  std::vector<uint32_t> bits{0,          0x80000000, 0x7f800000, 0xff800000,
                             1,          0x80000001, 0x7fc00001, 0x7f800001,
                             0xff800001, 0x7fffffff, 0x7f7fffff, 0xff7fffff,
                             0x00800000};
  std::vector<float> x(bits.size() * 4);
  for (size_t i = 0; i < x.size(); i++)
    std::memcpy(&x[i], &bits[i % bits.size()], 4);
  std::vector<uint64_t> shape{4, bits.size()};
  vtlc::Options o;
  o.tile_elements = 7;
  o.keyframe_interval = 2;
  for (int cycle = 0; cycle < 8; cycle++) {
    auto bytes = vtlc::compress(x, shape, o);
    auto y = vtlc::decompress(bytes);
    assert(!std::memcmp(x.data(), y.bits.data(), x.size() * 4));
    std::memcpy(x.data(), y.bits.data(), x.size() * 4);
  }
}
