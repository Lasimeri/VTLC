#pragma once
#include "codec.hpp"
#include <memory>
#include <optional>
namespace vtlc {
// Frames have fixed shape and contiguous row-major layout. Time is axis zero.
class StreamEncoder {
public:
  explicit StreamEncoder(std::span<const uint64_t> frame_shape,
                         Options options = {});
  ~StreamEncoder();
  StreamEncoder(StreamEncoder &&) noexcept;
  StreamEncoder &operator=(StreamEncoder &&) noexcept;
  void encode_frame(std::span<const float> frame);
  // Drain completed independent VTLC tensors during an unbounded stream.
  // After draining, use finish_chunks() rather than finish().
  std::optional<std::vector<uint8_t>> take_chunk();
  std::vector<std::vector<uint8_t>> finish_chunks();
  std::vector<uint8_t> finish();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
class StreamDecoder {
public:
  explicit StreamDecoder(std::span<const uint8_t> encoded,
                         Backend backend = Backend::Auto,
                         uint64_t memory_limit = uint64_t(1) << 30);
  ~StreamDecoder();
  StreamDecoder(StreamDecoder &&) noexcept;
  StreamDecoder &operator=(StreamDecoder &&) noexcept;
  bool decode_next(std::vector<uint32_t> &frame);
  size_t chunk_count() const;
  Tensor decode_chunk(size_t index) const;
  Tensor decode_all_parallel(unsigned threads = 0) const;
  uint64_t frame_count() const;
  std::span<const uint64_t> frame_shape() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace vtlc
