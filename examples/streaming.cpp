#include <bit>
#include <cassert>
#include <cstdint>
#include <vector>
#include <vtlc/streaming.hpp>
int main() {
  std::vector<uint64_t> frame_shape{3};
  vtlc::Options options;
  options.keyframe_interval = 2;
  vtlc::StreamEncoder encoder(frame_shape, options);
  std::vector<float> a{1, 2, 3}, b{2, 3, 4}, c{3, 4, 5};
  encoder.encode_frame(a);
  encoder.encode_frame(b);
  auto first = encoder.take_chunk();
  assert(first);
  auto chunk = vtlc::decompress(*first);
  assert(chunk.shape[0] == 2);
  encoder.encode_frame(c);
  auto remaining = encoder.finish_chunks();
  assert(remaining.size() == 1);
  auto last = vtlc::decompress(remaining[0]);
  for (size_t i = 0; i < c.size(); i++)
    assert(last.bits[i] == std::bit_cast<uint32_t>(c[i]));
}
