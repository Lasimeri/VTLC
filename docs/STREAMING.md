# Streaming and chunk access

`StreamEncoder` accepts contiguous FP32 frames of one fixed shape. Streaming uses a leading temporal axis; `Options::time_axis` must be zero. Each completed keyframe interval is encoded as a standalone VTLC tensor whose first frame is a keyframe. No chunk depends on earlier chunks, external state, or a dictionary.

There are two output patterns. Choose one for a given encoder:

1. **Finite envelope:** call `encode_frame` for each frame, then `finish()`. It returns one VTS1 byte vector containing the frame shape, frame count and all chunks. Compressed chunks are retained until finish.
2. **Continuous chunks:** after each `encode_frame`, call `take_chunk()` until it returns an empty optional. Store or transmit returned chunk byte vectors in order. At the end call `finish_chunks()` and process its remaining vectors. Each vector is independently decodable with whole-tensor `decompress()`.

Draining bounds the number of retained compressed chunks only when the caller drains often enough; a completed chunk is generated only when the configured interval fills, or at finish. The encoder retains the current uncompressed interval and codec scratch space. `finish()` cannot be used after any chunk has been drained, and neither finish method can be called twice.

```cpp
#include <vtlc/streaming.hpp>
#include <utility>
#include <vector>

int main() {
std::vector<uint64_t> frame_shape{3};
vtlc::Options options;
options.keyframe_interval = 2;
vtlc::StreamEncoder encoder(frame_shape, options);
std::vector<std::vector<uint8_t>> chunks;
for (int t = 0; t < 5; ++t) {
    std::vector<float> frame{float(t), 2.0f, 3.0f};
    encoder.encode_frame(frame);
    while (auto chunk = encoder.take_chunk())
        chunks.push_back(std::move(*chunk));
}
for (auto &chunk : encoder.finish_chunks())
    chunks.push_back(std::move(chunk));
for (const auto &chunk : chunks) {
    vtlc::Tensor frames = vtlc::decompress(chunk);
    // frames.shape begins with the number of frames in this chunk.
}
}
```

## Finite-stream decoder

Construct `StreamDecoder` from a VTS1 envelope produced by `finish()`. The decoder **borrows** the byte span; the input storage must remain alive and unchanged until all accesses complete. Construction checks the envelope, chunk metadata, and shape consistency; individual payloads are checked when decoded.

- `frame_count()` and `frame_shape()` describe the stream. The returned shape span borrows decoder storage.
- `decode_next(frame)` fills the caller's `vector<uint32_t>` with one frame's exact bit patterns and returns false at end of stream. It holds the current decoded chunk internally.
- `chunk_count()` reports independently decodable chunks; `decode_chunk(index)` returns an owning tensor and throws for an out-of-range index. Chunk offsets are scanned when the decoder is constructed.
- `decode_all_parallel(threads)` returns a single tensor with the time dimension first. Zero chooses a hardware-based count; the implementation caps workers at 32 plus the caller and uses a reusable pool. It allocates output and chunk scratch space and honors the logical-size check, not a total RSS cap.

Concurrent independent `decode_chunk` and `decode_all_parallel` calls on a decoder are exercised in the test suite. Do not concurrently call the stateful `decode_next` on the same decoder without external synchronization. Every compressed byte span must remain valid throughout concurrent work.

The VTS1 envelope and standalone VTLC chunks are distinct formats. An application using continuous chunks must preserve their order and any framing needed by its transport; a sequence of raw chunk byte vectors is not itself a VTS1 envelope. See [FORMAT.md](FORMAT.md) for field layouts.
