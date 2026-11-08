#include "vtlc/codec.hpp"
#include "vtlc/streaming.hpp"
#include <fstream>
#include <vector>
static void save(const char *name, const std::vector<uint8_t> &x) {
  std::ofstream f(name, std::ios::binary);
  f.write((const char *)x.data(), x.size());
}
int main() {
  std::vector<uint64_t> shape{4, 33};
  std::vector<float> input(132);
  for (int i = 0; i < 132; i++)
    input[i] = float(i % 33) * 0.1f + float(i / 33) * 0.001f;
  vtlc::Options o;
  o.preset = vtlc::Preset::Ratio;
  o.checksum = false;
  save("fuzz/corpus/tensor", vtlc::compress(input, shape, o));
  vtlc::StreamEncoder stream(std::span<const uint64_t>(shape).subspan(1), o);
  for (int i = 0; i < 4; i++)
    stream.encode_frame(std::span<const float>(input).subspan(i * 33, 33));
  save("fuzz/corpus/stream", stream.finish());
}
