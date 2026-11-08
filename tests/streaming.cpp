#include "vtlc/streaming.hpp"
#include <cassert>
#include <cstring>
#include <future>
#include <random>
int main() {
  std::vector<uint64_t> s{17};
  vtlc::Options o;
  o.keyframe_interval = 3;
  o.tile_elements = 7;
  vtlc::StreamEncoder enc(s, o);
  std::vector<float> all(17 * 10);
  std::mt19937 gen(77);
  for (auto &f : all) {
    uint32_t b = gen();
    std::memcpy(&f, &b, 4);
  }
  for (int i = 0; i < 10; i++)
    enc.encode_frame(std::span<const float>(all).subspan(i * 17, 17));
  auto bytes = enc.finish();
  vtlc::StreamDecoder dec(bytes);
  assert(dec.chunk_count() == 4 && dec.frame_count() == 10);
  auto parallel = dec.decode_all_parallel(3);
  assert(!std::memcmp(parallel.bits.data(), all.data(), all.size() * 4));
  std::vector<std::future<vtlc::Tensor>> groups;
  for (int j = 0; j < 6; j++)
    groups.push_back(std::async(std::launch::async,
                                [&dec] { return dec.decode_all_parallel(3); }));
  for (auto &g : groups) {
    auto y = g.get();
    assert(!std::memcmp(y.bits.data(), all.data(), all.size() * 4));
  }
  std::vector<std::future<vtlc::Tensor>> tasks;
  for (size_t i = 0; i < dec.chunk_count(); i++)
    tasks.push_back(std::async(std::launch::async,
                               [&dec, i] { return dec.decode_chunk(i); }));
  for (size_t i = 0; i < tasks.size(); i++) {
    auto t = tasks[i].get();
    assert(!std::memcmp(t.bits.data(), all.data() + i * 3 * 17,
                        t.bits.size() * 4));
  }
  std::vector<uint32_t> frame;
  for (int i = 0; i < 10; i++) {
    assert(dec.decode_next(frame));
    assert(!std::memcmp(frame.data(), all.data() + i * 17, 17 * 4));
  }
  assert(!dec.decode_next(frame));
  vtlc::StreamEncoder continuous(s, o);
  size_t at = 0;
  for (int i = 0; i < 10; i++) {
    continuous.encode_frame(std::span<const float>(all).subspan(i * 17, 17));
    while (auto c = continuous.take_chunk()) {
      auto t = vtlc::decompress(*c);
      assert(!std::memcmp(t.bits.data(), all.data() + at, t.bits.size() * 4));
      at += t.bits.size();
    }
  }
  for (auto &c : continuous.finish_chunks()) {
    auto t = vtlc::decompress(c);
    assert(!std::memcmp(t.bits.data(), all.data() + at, t.bits.size() * 4));
    at += t.bits.size();
  }
  assert(at == all.size());
  std::vector<uint64_t> empty_shape{0};
  vtlc::StreamEncoder empty_encoder(empty_shape, o);
  for (int i = 0; i < 5; ++i)
    empty_encoder.encode_frame({});
  auto empty_bytes = empty_encoder.finish();
  vtlc::StreamDecoder empty_decoder(empty_bytes);
  assert(empty_decoder.frame_count() == 5);
  for (int i = 0; i < 5; ++i) {
    assert(empty_decoder.decode_next(frame));
    assert(frame.empty());
  }
  assert(!empty_decoder.decode_next(frame));
  auto empty_tensor = empty_decoder.decode_all_parallel(2);
  assert(empty_tensor.bits.empty() && empty_tensor.shape[0] == 5);
  for (size_t n = 0; n < bytes.size(); n++) {
    try {
      vtlc::StreamDecoder bad(std::span<const uint8_t>(bytes).first(n));
      assert(false);
    } catch (const std::exception &) {
    }
  }
}
