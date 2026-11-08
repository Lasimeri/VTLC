#include "vtlc/codec.hpp"
#include <cassert>
#include <cstring>
#include <zstd.h>
using Bytes = std::vector<uint8_t>;
static void put(Bytes &b, uint64_t x, int n) {
  for (int i = 0; i < n; i++)
    b.push_back(uint8_t(x >> (i * 8)));
}
static Bytes container(unsigned codec, unsigned count, const Bytes &data) {
  Bytes b;
  put(b, 0x434c5456, 4);
  put(b, 1, 2);
  put(b, 1, 1);
  put(b, 1, 1);
  put(b, 2, 4);
  put(b, 0, 4);
  put(b, 8192, 4);
  put(b, 16, 4);
  put(b, count, 8);
  put(b, 1, 8);
  put(b, count, 8);
  put(b, 0, 1);
  put(b, codec, 1);
  put(b, 0, 2);
  put(b, count, 4);
  put(b, data.size(), 8);
  b.insert(b.end(), data.begin(), data.end());
  return b;
}
struct Bits {
  Bytes b;
  unsigned pos = 0;
  void write(unsigned x, unsigned n) {
    for (unsigned i = 0; i < n; i++) {
      if (!(pos % 8))
        b.push_back(0);
      b.back() |= ((x >> i) & 1) << (pos % 8);
      pos++;
    }
  }
};
int main() {
  std::vector<uint32_t> words{0, 0, 42, 0};
  Bytes runs{4, 3, 42, 0, 0, 0, 2};
  assert(vtlc::decompress(container(4, 4, runs)).bits == words);
  Bits w;
  w.write(0, 1);
  w.write(1, 1);
  w.write(1, 1);
  w.write(28, 5);
  w.write(0, 5);
  w.write(1, 1);
  w.write(1, 1);
  w.write(0, 1);
  w.write(1, 1);
  w.write(1, 1);
  w.write(1, 1);
  w.write(31, 5);
  w.write(0, 5);
  w.write(1, 1);
  assert((vtlc::decompress(container(5, 4, w.b)).bits ==
          std::vector<uint32_t>{0, 8, 8, 1}));
  std::vector<uint32_t> original{1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto sh = vtlc::bitshuffle_words(original);
  Bytes z(ZSTD_compressBound(sh.size()));
  auto n = ZSTD_compress(z.data(), z.size(), sh.data(), sh.size(), 1);
  assert(!ZSTD_isError(n));
  z.resize(n);
  assert(vtlc::decompress(container(3, original.size(), z)).bits == original);
  std::vector<float> values(100, 1.f);
  std::vector<uint64_t> shape{2, 50};
  auto encoded = vtlc::compress(values, shape);
  encoded.back() ^= 1;
  try {
    (void)vtlc::decompress(encoded);
    assert(false);
  } catch (const std::exception &) {
  }
}
