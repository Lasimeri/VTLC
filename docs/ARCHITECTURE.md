# Architecture

The whole-tensor encoder reads caller-owned FP32 storage by span, gathers each logical time frame into integer words, and builds tile residuals with zero, previous-frame, modular extrapolation or fused extrapolation. Fast, Balanced and Adaptive select predictors using sampled residual costs; Ratio evaluates all predictors. Candidate encodings are compared by actual payload size. CRC and tile metadata are serialized with each selected payload. The decoder reverses the chosen encoding, predicts from reconstructed history, XORs, and scatters words into row-major output. A new keyframe interval resets legal history. There is no external dictionary or reference state.

Backend selection uses compiler-provided CPU feature detection, which accounts for OS extended-register support. SIMD functions use local target attributes; generic code does not require AVX. Fused arithmetic is constrained by MXCSR save/set/restore and exceptional-value fallback. Integer predictors handle all representations unchanged.

Streaming holds at most one uncompressed interval when completed chunks are drained. Each chunk is a standalone tensor. Finite VTS1 envelopes add frame shape and chunk boundaries. A decoder scans that envelope into chunk spans and can decode chunks independently with a reusable bounded thread pool.

Costs include full frame gather/scatter and candidate vectors, repeated predictor and entropy work, scalar residual packing and inverse bitshuffle, CRC, and Zstandard contexts. Memory limits currently constrain logical uncompressed size rather than all allocations.
