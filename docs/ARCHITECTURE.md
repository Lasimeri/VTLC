# Architecture

## Exact reconstruction

For a frame element, the encoder reads the FP32 representation as `uint32_t` bits `u`, computes predicted bits `p`, and stores the complete residual `r = u XOR p`. The decoder computes the same `p` from already reconstructed history and obtains `u = r XOR p`. XOR retains every bit of prediction error; predictor accuracy affects size, not the reconstructed representation. Signaling NaNs and payloads remain integer data except when a predictor's explicitly checked finite normal operands are used for FMA.

## Tensor traversal and history

The whole-tensor encoder accepts caller-owned row-major FP32 storage by span. It gathers one logical time frame into aligned integer-word buffers; when the time axis is not zero, it uses the dimensions following the axis to map between temporal and original ordering. Three frame buffers are reused for current and two previous frames. The decoder reconstructs a frame and scatters it into the original row-major layout.

A keyframe interval resets predictor eligibility. At its first frame, only zero prediction is legal. At the next frame, zero and previous-frame prediction are legal. Later frames can also use modular and fused extrapolation. Each frame is divided into independently encoded tiles, although tiles in later frames depend on prior reconstructed frames in their interval.

| ID | Prediction | Notes |
|---:|---|---|
| 0 | Zero bits | Works without history. |
| 1 | Previous word | Integer copy, including exceptional FP32 bits. |
| 2 | `2 * previous - previous2` | Unsigned modulo 2^32. |
| 3 | `fma(2.0f, previous, -previous2)` | Single fused rounding; non-normal operands or result fall back to P1. |

On x86, P3 saves MXCSR, sets round-to-nearest-even with FTZ/DAZ disabled and exceptions masked during prediction, and restores the caller's value. Scalar uses `std::fma`; supported vector paths use FMA intrinsics. The encoder and decoder both apply the same rules. Non-x86 decoding rejects P3 semantics. See [FORMAT.md](FORMAT.md) for the serialized rules.

## Selection and residual coding

Fast, Balanced and Adaptive estimate predictor costs by sampling residuals at a fixed stride; Ratio evaluates all legal predictors. A forced predictor bypasses sampling, subject to history availability. For each evaluated predictor, the implementation compares actual payload sizes from the permitted modes and records the chosen predictor and codec in the tile record. The raw residual mode ensures the chosen tile payload need not exceed raw residual bytes, although headers and checksums can make the overall container larger than the original.

The available modes are raw little-endian residuals, 128-word block bit packing, zero/literal runs, Gorilla-inspired significant-bit windows, Zstandard, and bitshuffle followed by Zstandard. Not every preset evaluates every mode. SIMD accelerates zero scans, bit-plane transpose and inverse, width statistics, and 8/16-bit packing on AVX2. Other packing widths use the generic implementation. Zstandard is optional at build time.

## Dispatch and streaming

Compiler-provided CPU feature detection selects scalar, AVX2, AVX2+FMA or AVX-512F at runtime, with function-local target attributes rather than global AVX flags. CPU and OS support must be available for a forced backend. AVX-512F has not been tested on the recorded machine. Integer AVX2 prediction has aligned and unaligned paths; vector tails use scalar code.

For streaming, `StreamEncoder` groups frames into standalone VTLC tensors. Drain completed chunks during continuous input or finish a finite VTS1 envelope. `StreamDecoder` scans envelope records into chunk spans, can decode sequential frames, and uses a bounded reusable worker pool for parallel decoding of independent chunks. Chunk references borrow the envelope bytes. See [STREAMING.md](STREAMING.md).

## Cost and limits

The codec allocates frame buffers and evaluates candidate residual representations. Gather/scatter, temporary candidate vectors, residual coding, CRC, and Zstandard contexts can dominate the predictor kernel. Logical tensor-size limits do not cap all allocations. The implementation does not authenticate the whole container, and raw fallback cannot eliminate metadata overhead. Measurements and reproducibility limits are documented in [BENCHMARKS.md](BENCHMARKS.md); operational caveats are in [SECURITY.md](SECURITY.md).
