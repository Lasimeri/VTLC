# Integrity and operational limits

VTLC is research-grade code. Do not rely on it for production data integrity or hostile inputs without additional review, isolation, and validation. Successful tests and fuzz runs cover only the tested cases.

## Checksums and corruption

Checksums are enabled by default. Each checksum is CRC-32/ISO-HDLC over the tile predictor, codec, element count, and compressed payload. A checksum mismatch is rejected during decoding. CRC is designed for accidental corruption detection; it is not a cryptographic MAC or signature. It does not cover the complete container header or VTS1 envelope. A changed header, a disabled checksum, a collision, or an intentionally crafted stream can therefore evade the checksum. Applications that need authenticated storage or transport must protect the complete serialized byte sequence with an appropriate authenticated mechanism outside VTLC.

The decoder checks the magic, version, dtype, byte-order identifier, rank, shape product, supported modes, keyframe history, lengths, padding and end of input. Zstandard output must match the expected tile size. Unsupported or malformed content generally raises an exception. These checks and the logical memory limit are useful defenses, but are not a security boundary or a guarantee that all corruption will be detected.

## Memory limits

`Options::memory_limit` and the corresponding decoder argument limit the logical uncompressed tensor byte count. The default is 1 GiB. They do **not** strictly limit peak process memory. Input and output storage, three frame-history buffers, temporary candidate/residual buffers, compressed output, Zstandard allocations, and parallel chunk decoding can all contribute to peak use. Multiple concurrent operations add their own memory requirements. Finite `StreamEncoder::finish()` retains compressed chunks before creating the envelope; drain chunks for continuous input when appropriate.

Use a separate process, OS resource limits, and an appropriate deployment policy when a hard process-memory boundary is needed. A byte limit also does not bound CPU time for every possible input or eliminate allocation failure. Catch errors and treat failed decodes as unusable data.

## Format and compatibility

The container is versioned and little-endian; it stores the shape and predictor and codec identifiers. No external reference tensors or dictionaries are needed. A build without Zstandard rejects Zstandard-coded tiles. Predictor 3 requires the documented fused semantics; non-x86 decoding explicitly rejects it. A decoder rejects unsupported modes rather than silently interpreting them as another codec.

`inspect()` validates the header and declared logical size only. It does not validate the rest of the stream. `StreamDecoder` borrows its input, which must not be freed or modified while the decoder or its concurrent chunk calls are using it. Keep an independently verified backup for valuable data; neither the codec's checksum nor an encode/decode test is a substitute for a backup strategy.

## Responsible deployment

The encoder's ratio and throughput are data-dependent. Incompressible data can expand because of metadata even when every tile uses raw fallback. CPU feature dispatch depends on the compiler's runtime detection and operating-system register-state support; never force a backend on unsupported hardware. AVX-512F has not been executed on the recorded development machine. Review [FORMAT.md](FORMAT.md), [TESTING.md](TESTING.md), and [BENCHMARKS.md](BENCHMARKS.md) before making compatibility or performance claims.
