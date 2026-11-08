# Testing and validation

All correctness comparisons must use the original 32-bit representations. Ordinary FP32 equality is unsuitable for NaNs and does not distinguish positive and negative zero.

## Test suite

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The CMake test executables keep assertions enabled even in a release build. The suite includes:

| CTest name | Coverage |
|---|---|
| `roundtrip` | Randomized tensors, axes and available backends; truncation and deterministic byte mutations. |
| `edge_cases` | Explicit exceptional FP32 representations and repeated cycles. |
| `streaming` | Sequential, chunk, continuous and parallel decoding, plus truncated streams. |
| `fused_prediction` | Forced P3, backend agreement, exceptional bit patterns and caller rounding modes. |
| `bitshuffle` | Scalar and available SIMD transforms, tails and inverse. |
| `residual_codecs` | Constructed residual modes and checksum rejection; Zstandard build only. |
| `ratio_roundtrip` | Presets and forced predictors over deterministic input families. |
| `c_interface` | C linking, ownership and round trip. |
| `vector_packing` | 8/16-bit packed widths, tails and backend agreement. |
| `alignment` | Aligned and unaligned predictor inputs. |

Backend tests skip unsupported hardware. A passing suite on an AVX2 machine does not validate AVX-512 execution. The tests do not establish universal correctness, production readiness, or robustness against all malicious streams.

## Sanitizers

```sh
cmake -S . -B build-sanitize -DCMAKE_BUILD_TYPE=Debug -DVTLC_SANITIZE=ON
cmake --build build-sanitize -j
ctest --test-dir build-sanitize --output-on-failure
```

This enables AddressSanitizer and UndefinedBehaviorSanitizer. Use a compiler and runtime that support those options. Sanitizers can change performance and memory use; do not use their timings as release benchmarks.

## Coverage-guided fuzzing

Clang is required for the libFuzzer target. The following uses copies of the checked-in seed corpus so the run does not modify tracked corpus files:

```sh
cmake -S . -B build-fuzz -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Debug -DVTLC_SANITIZE=ON -DVTLC_FUZZ=ON
cmake --build build-fuzz --target vtlc_fuzz -j
mkdir -p build-fuzz/corpus
cp fuzz/corpus/tensor fuzz/corpus/stream build-fuzz/corpus/
./build-fuzz/vtlc_fuzz build-fuzz/corpus -runs=100000 -max_len=8192
```

The harness exercises whole-tensor and finite-stream decoders with a 1 MiB logical tensor limit. A short local 100,000-input run completed without a reported sanitizer failure; this is a bounded sample, not a proof of security. Keep crashing inputs and reproduction commands when extending the campaign. Full-header authentication, strict process-memory limits and comprehensive adversarial validation remain outside the implementation's present guarantees.

## Reproducing SIMD checks

Use `vtlc::available` before forcing a backend. Compare compressed bytes and decoded `uint32_t` words against scalar for the same input and options. Include lanes containing signed zeros, subnormals, infinities, signaling and quiet NaNs, payload variations, and values that make fused extrapolation non-normal. For fused prediction, test caller rounding and FTZ/DAZ settings and verify MXCSR is restored. AVX-512F needs independent execution on suitable hardware; compilation alone is insufficient.
