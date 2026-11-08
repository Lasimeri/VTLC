# VTLC

**Vectorized Temporal Lossless Compression** is a C++20 codec for sequences of FP32 tensors. It predicts values across time and stores the complete XOR residual, preserving every original 32-bit pattern—including signed zeros and NaN payloads.

> **Status:** Research-grade software, not yet production-ready. See [Limitations](#limitations) before using it for important or untrusted data.

- **Self-contained:** compressed data includes its shape and decoding metadata.
- **Vectorized:** runtime dispatch supports scalar, AVX2/FMA and AVX-512F.
- **Streamable:** independent chunks support sequential and parallel decoding.
- **C and C++ APIs:** Zstandard is optional; no Python runtime is required.

## Quick start

Requirements: CMake 3.16+, a C++20 compiler, and Zstandard development headers and library. To build without Zstandard, add `-DVTLC_ZSTD=OFF` to the configure command.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Compress a contiguous row-major tensor and compare the restored **bits**:

```cpp
#include <vtlc/codec.hpp>
#include <bit>
#include <cassert>
#include <cstdint>
#include <vector>

int main() {
    std::vector<float> input{1, 2, 3, 1, 2, 3};
    std::vector<uint64_t> shape{2, 3};  // 2 time steps, 3 values each

    auto compressed = vtlc::compress(input, shape);
    auto restored = vtlc::decompress(compressed);

    for (size_t i = 0; i < input.size(); ++i)
        assert(restored.bits[i] == std::bit_cast<uint32_t>(input[i]));
}
```

The decoder returns the shape and `uint32_t` bit patterns. The whole-tensor API supports any time axis; streaming uses the leading axis. See the [API reference](docs/API.md) and [streaming guide](docs/STREAMING.md) for options, ownership, and C examples.

## How it works

```text
  FP32 tensor + shape
          |
          v
  +---------------------+
  | Frames and tiles    |<---- keyframes reset prediction history
  +----------+----------+
             |
             v
  +---------------------+      scalar / AVX2 / AVX-512F
  | Predict, then XOR   |<---- runtime CPU dispatch
  +----------+----------+
             |
             v
  +---------------------+
  | Encode residuals    |      raw, bit-packed, zero runs,
  | and select a mode   |      Gorilla-style, or Zstandard modes
  +----------+----------+
             |
             v
  +---------------------+
  | Self-contained VTLC |----> decode, predict, XOR, restore bits
  | container + metadata|
  +---------------------+

  Streaming: frames -> independent chunks -> sequential / parallel decode
```

For each value, the residual is `original_bits XOR predicted_bits`. The decoder computes the same prediction and XORs it with the residual, recovering the original bits exactly. Prediction and encoding choices affect size and speed, not losslessness. See [Architecture](docs/ARCHITECTURE.md) for predictor and codec details.

## Benchmarks

Run the end-to-end and predictor-only benchmarks locally:

```sh
./build/vtlc_bench
./build/vtlc_kernels
```

On the recorded Ryzen 7 5800X run, synthetic smooth data compressed to **17.4%** of its original size, while noisy correlated data compressed to **79.5%**. Random bit patterns expanded slightly to **100.1%** because of metadata. These results are dataset-dependent; a 30–50% reduction is not guaranteed.

The [benchmark guide](docs/BENCHMARKS.md) explains the methodology and limitations. Full results, including Zstandard comparisons, are in [benchmark-results.csv](docs/benchmark-results.csv).

## Limitations

- **Memory:** the configured limit applies to logical uncompressed tensor size, not peak process memory.
- **Integrity:** optional tile CRCs do not authenticate the container or protect its entire header. Do not rely on VTLC for hostile inputs or production data integrity.
- **Hardware:** AVX-512F is implemented but has not been executed on the recorded development machine.
- **Performance:** the current encoder can be slower than plain Zstandard; results vary by data and configuration.

See [Security and operational limits](docs/SECURITY.md) and [Testing](docs/TESTING.md) for details.

## Documentation

| Guide | Contents |
|---|---|
| [Build and install](docs/BUILD.md) | Dependencies, build options, linking, and ISA support |
| [API reference](docs/API.md) | C++ and C interfaces, presets, and ownership |
| [Streaming](docs/STREAMING.md) | Frame input, independent chunks, and parallel decoding |
| [Architecture](docs/ARCHITECTURE.md) | Predictors, residual modes, dispatch, and costs |
| [Format](docs/FORMAT.md) | VTLC tensor and VTS1 stream layouts |
| [Testing](docs/TESTING.md) | Tests, sanitizers, and fuzzing |
| [Benchmarks](docs/BENCHMARKS.md) | Methodology, comparisons, and measured results |
| [Security](docs/SECURITY.md) | Integrity, memory, compatibility, and deployment limits |

## License and influences

VTLC is available under the [MIT license](LICENSE). Its design draws on ideas from [Gorilla](https://dl.acm.org/doi/10.14778/2824032.2824078), [Sprintz](https://arxiv.org/abs/1808.02515), and [Bitshuffle](https://github.com/kiyo-masui/bitshuffle). Zstandard is an optional external dependency with its own license.
