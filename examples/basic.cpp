#include <bit>
#include <cassert>
#include <cstdint>
#include <vector>
#include <vtlc/codec.hpp>
int main() {
  std::vector<float> x{1, 2, 3, 1, 2, 3};
  std::vector<uint64_t> shape{2, 3};
  auto bytes = vtlc::compress(x, shape);
  auto restored = vtlc::decompress(bytes);
  for (size_t i = 0; i < x.size(); ++i)
    assert(std::bit_cast<uint32_t>(x[i]) == restored.bits[i]);
}
