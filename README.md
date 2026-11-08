# VTLC

VTLC is a C++20 bitwise-lossless temporal FP32 tensor codec with a C interface. It implements integer and fused floating-point predictors, XOR residuals, several lossless residual encodings, independent streaming chunks, and runtime scalar, AVX2/FMA and AVX-512F dispatch. This is a research-grade implementation, **not yet production-ready**: see limitations below. No Python is used.

## Architecture

```text
  Contiguous FP32 tensor + shape + options
                    |
                    v
       +---------------------------+
       | Gather frames, split tiles|
       | Reset history at keyframes|
       +-------------+-------------+
                     |
                     v
       +---------------------------+       scalar / AVX2 / AVX-512F
       | Choose temporal predictor |<------ runtime CPU dispatch
       | XOR prediction with bits  |
       +-------------+-------------+
                     |
                     v
       +---------------------------+
       | Raw / packed / zero runs  |
       | Gorilla / bitshuffle+zstd |
       | Select an evaluated mode  |
       +-------------+-------------+
                     |
                     v
       +---------------------------+
       | VTLC header + tile records|
       | optional CRC + payloads   |
       +-------------+-------------+
                     |
                     v
       +---------------------------+
       | Decode residuals, predict,|
       | XOR, and scatter FP32 bits|
       +-------------+-------------+
                     |
                     v
          Original shape + FP32 bits

  Streaming: frames -> independent VTLC chunks -> sequential or parallel decode
```

The history used by a tile comes from earlier frames within the current keyframe interval. Each streaming chunk is a standalone VTLC tensor. The exact bitwise invariant and the implementation's resource and integrity limits are described in the documentation below.

## Documentation

- [Build, installation, and platform support](docs/BUILD.md)
- [C++ and C API reference](docs/API.md)
- [Streaming and chunk access guide](docs/STREAMING.md)
- [Architecture and predictor behavior](docs/ARCHITECTURE.md)
- [VTLC and VTS1 binary formats](docs/FORMAT.md)
- [Testing, sanitizers, and fuzzing](docs/TESTING.md)
- [Benchmark methodology and recorded results](docs/BENCHMARKS.md)
- [Integrity, memory, and deployment limitations](docs/SECURITY.md)

## Build and run

Requires CMake 3.16+ and a C++20 compiler. Zstandard development headers and library are enabled by default; set `-DVTLC_ZSTD=OFF` to build without it. Such builds reject containers containing Zstandard tiles.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/vtlc_bench
./build/vtlc_kernels
cmake --install build --prefix "$PWD/install"
```

The C++ headers are `include/vtlc/codec.hpp` and `include/vtlc/streaming.hpp`; the C interface is `include/vtlc/c_api.h`. See `examples/basic.cpp`. Input must be contiguous row-major FP32; the whole-tensor API supports any time axis. Output is a self-contained byte vector. The decoder returns `uint32_t` bit patterns and shape, avoiding floating-point arithmetic on signaling NaNs. Buffers may be empty; rank must be 1–32.

Fast, Balanced and Adaptive sample residual costs to select a legal predictor per tile; Ratio evaluates all legal predictors. Each then considers raw, per-128-word fixed-width packing, and zero-run modes. Presets other than Fast also try Zstandard. Ratio additionally tries Gorilla-style significant windows and AVX2 bitshuffle plus Zstandard. Adaptive skips Zstandard when sampled residuals appear costly to compress. The smallest evaluated payload wins. Raw tile fallback prevents payload expansion for that tile, but metadata can still increase total size. Checksums default on and cover tile metadata and payload, not the tensor header.

The fused predictor executes under nearest-even, FTZ/DAZ disabled, restoring caller x86 MXCSR. Non-normal operands and results fall back to the previous word; signaling NaNs are not evaluated in floating point. Scalar uses `std::fma`; AVX2/FMA and AVX-512F use fused vector intrinsics. Runtime CPU detection includes OS support for extended register state. No global AVX flags are used. Forced unavailable backends throw. AVX-512F code does not require AVX-512BW, DQ or VL. AVX-512F has not been executed on the development machine.

`StreamEncoder` accepts fixed-shape frames. Drain `take_chunk()` periodically to keep compressed output from accumulating; each returned VTLC chunk is independently decodable with `decompress()`. `finish_chunks()` yields remaining chunks. Alternatively `finish()` produces one self-contained VTS1 envelope, buffering all compressed chunks. `StreamDecoder` supports `decode_next()`, `decode_chunk()`, and `decode_all_parallel()`. It borrows the source buffer, which must outlive the decoder. Streaming time axis is zero. Parallel decoding uses a reusable pool of up to 32 worker threads plus the calling thread.

## Validation and limitations

The project has randomized and deterministic bitwise round trips, cross-backend tests, special-value tests, malformed/truncated input checks, codec tests, repeated cycles, concurrency tests, sanitizer builds, and a libFuzzer target. Coverage is finite and does not prove absence of bugs. The logical uncompressed tensor byte limit is configurable, but it does **not** bound peak process memory: scratch buffers, compressed output, Zstandard and parallel workers add overhead. No integrity protection covers the entire header, and CRC is not authentication. Do not rely on this implementation for production data integrity or hostile input.

Further work includes independent AVX-512 hardware validation, packing beyond the 8- and 16-bit SIMD fast paths and improved inverse bitshuffle performance, full peak-memory limits, broad real-world benchmark data, improved adaptive predictor cost models. The integer AVX2 predictor has aligned and unaligned paths; FMA kernels use unaligned loads and scalar tails. The current encoder evaluates and allocates many candidate buffers, and is slower than plain Zstandard in the supplied synthetic runs. The 30–50% reduction objective is dataset-dependent and not guaranteed. See [benchmark notes](docs/BENCHMARKS.md) and [format](docs/FORMAT.md).

The XOR residual is related to [Gorilla](https://dl.acm.org/doi/10.14778/2824032.2824078); adaptive prediction and packing are related to [Sprintz](https://arxiv.org/abs/1808.02515). [Bitshuffle](https://github.com/kiyo-masui/bitshuffle), [Intel's Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html), and [RFC 8878](https://www.rfc-editor.org/rfc/rfc8878.html) were consulted. Zstandard is an external dependency with its own license; no third-party source code is copied here.
