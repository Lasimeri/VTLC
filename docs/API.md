# API reference

The C++ API is declared in `vtlc/codec.hpp` and `vtlc/streaming.hpp`; the C interface is in `vtlc/c_api.h`. Operations report errors by throwing `std::exception` subclasses in C++, or by returning a nonzero status through the C interface. Compressed bytes contain the shape and decoding metadata; callers do not provide a reference tensor or dictionary.

## Whole-tensor C++ API

```cpp
#include <vtlc/codec.hpp>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <vector>

int main() {
std::vector<float> input{1, 2, 3, 1, 2, 3};
std::vector<uint64_t> shape{2, 3};
vtlc::Options options;
options.time_axis = 0;
auto encoded = vtlc::compress(input, shape, options);
auto restored = vtlc::decompress(encoded);
for (size_t i = 0; i < input.size(); ++i) {
    if (restored.bits[i] != std::bit_cast<uint32_t>(input[i]))
        throw std::runtime_error("bit mismatch");
}
}
```

The input span is borrowed for the duration of `compress`; it must point to contiguous row-major `float` values. The shape is rank 1–32, its product must equal the span length, and the configured axis must be within the rank. Empty dimensions are supported. `compress` returns an owning byte vector. The input is accepted without a whole-tensor input copy, but the implementation gathers frames and allocates scratch space.

`decompress(bytes, backend, memory_limit)` returns a `Tensor` owning `shape` and `bits`. Each `bits` element is the original 32-bit FP32 representation in row-major order. Use a bit-preserving copy or `std::bit_cast<float>` where floating values are needed; ordinary floating-point equality cannot validate NaN payloads or distinguish signed zeros. The default limit is 1 GiB of logical uncompressed tensor data, not a cap on all memory used during the call.

`inspect(bytes, memory_limit)` returns header metadata: shape, axis, tile size, keyframe interval, and element count. It parses the header only; successful inspection does **not** validate all tiles, payloads, or checksums. Use `decompress` to decode and validate the full stream.

## Options

| Field | Default | Meaning |
|---|---:|---|
| `time_axis` | `0` | Temporal dimension in the whole-tensor shape. |
| `tile_elements` | `8192` | Maximum number of elements in one encoded tile; must be nonzero. |
| `keyframe_interval` | `16` | Frames between predictor resets; must be nonzero. |
| `preset` | `Balanced` | Select candidate evaluation and entropy-coding behavior. |
| `backend` | `Auto` | Runtime SIMD backend or forced reference backend. |
| `predictor` | `-1` | Automatic selection, or force 0–3 when sufficient history is available. |
| `checksum` | `true` | Write a CRC for each tile's selected metadata and payload. |
| `memory_limit` | `1 << 30` | Logical uncompressed tensor byte limit; not total RSS. |

When a predictor is forced, the encoder uses `min(requested, legal maximum)` for frames without enough history: predictor 0 at a keyframe and at most predictor 1 in the next frame. The preset remains lossless regardless of selection. Fast excludes Zstandard and uses sampled prediction selection; Balanced also evaluates Zstandard; Adaptive may skip Zstandard when sampled residuals suggest little benefit; Ratio evaluates all legal predictors and additional residual modes.

`available(backend)` checks support. `Auto` chooses the best detected backend. `AVX2` uses the scalar fused implementation for P3, while `AVX2_FMA` and `AVX512F` use vector fused operations. Forced unavailable backends throw. Predictor 3 is rejected on non-x86 builds; automatic encoding there does not select it.

## Low-level transforms

`predict_xor_words(input, previous, previous2, output, predictor, backend)` computes prediction XOR input; applying the same transform with identical history reverses it. Output length must equal input length, and the predictor's required histories must have that length. `bitshuffle_words` and `bitunshuffle_words` are reversible word transforms; unshuffle requires the exact original element count. These helpers expose codec primitives, not independently versioned containers.

## C interface and ownership

```c
#include <vtlc/c_api.h>
#include <stdint.h>
#include <string.h>

int main(void) {
float input[] = {1, 2, 3, 1, 2, 3};
uint64_t shape[] = {2, 3}, compressed_size = 0, count = 0;
uint64_t *decoded_shape = 0;
unsigned char *compressed = 0, *decoded = 0;
unsigned rank = 0;
if (vtlc_compress_c(input, shape, 2, 0, &compressed, &compressed_size))
    return 1;
if (vtlc_decompress_c(compressed, compressed_size, &decoded, &count,
                      &decoded_shape, &rank)) {
    vtlc_free(compressed);
    return 1;
}
int mismatch = count != 6 || memcmp(input, decoded, sizeof input) != 0;
vtlc_free(compressed);
vtlc_free(decoded);
vtlc_free(decoded_shape);
return mismatch;
}
```

Successful calls return zero. Error details are available through `vtlc_error()`; the returned text is thread-local and may change on a later error in that thread. Free all three returned allocations using `vtlc_free`, including the decoded shape. The C interface uses default compression options except for temporal axis and does not expose all C++ settings. The output byte buffer holds native `uint32_t` representations of the FP32 words; the container itself is little-endian as described in [FORMAT.md](FORMAT.md).

See [STREAMING.md](STREAMING.md) for frame-by-frame and chunk APIs, lifetime constraints, and parallelism.
