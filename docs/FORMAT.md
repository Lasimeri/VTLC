# VTLC format version 1

All integers are unsigned little-endian. The header contains four bytes ASCII `VTLC`; u16 version 1; u8 dtype 1 (FP32); u8 byte-order 1 (little endian); u32 rank, temporal axis, tile elements, keyframe interval; u64 element count; and rank u64 dimensions. Rank is 1–32, shape product equals element count. Tensor ordering is row-major.

Frames are serialized in time order, with each frame in row-major order excluding the temporal axis. Each tile header has u8 predictor, u8 codec, u16 flags, u32 element count and u64 payload length. Flag 0 means no checksum; flag 1 adds a u32 CRC-32/ISO-HDLC before payload. CRC covers predictor and codec bytes, little-endian element count, and payload; it does not cover tensor header fields and is not authentication.

Predictor 0 is zero; 1 is the previous-frame word; 2 is twice the previous word minus the second previous word modulo 2^32; 3 is fused `fma(2.0f, previous, -previous2)`, rounded nearest-even with FTZ and DAZ disabled. When either operand or the result is not a finite normal FP32 value, predictor 3 uses the previous word. The caller's x86 MXCSR is restored. Non-x86 builds reject predictor 3. At a keyframe only predictor 0 is valid; the second frame allows 0 or 1. The residual is actual XOR predictor.

Codec 0 is raw little-endian u32 residuals. Codec 1 packs blocks of at most 128 words: one byte width 0–32, followed by width bits per word, least-significant bit first, rounded to a byte boundary per block. Codec 2 is a Zstandard frame decoded to little-endian u32 residuals. Codec 3 is Zstandard of bitshuffled words: for each group of eight words, 32 bytes hold successive bit positions, with bit k of each byte from word k; remaining words are little-endian raw. Codec 4 contains unsigned base-128 varint runs: token is `(count<<1)|literal`, followed by count little-endian u32 values if literal is 1. Codec 5 is a little-endian bitstream: zero uses control bit 0; nonzero uses 1 and then a reuse flag (0 reuse, 1 new). A new window contains 5-bit leading-zero count, 5-bit significant-width-minus-one, then significant bits. Reuse emits the previous window's width; bit fields are least-significant bit first. Padding must be zero. Zstandard outputs must have exactly 4 bytes per residual; dictionaries are not used.

The decoder rejects trailing data, invalid widths, predictor histories and lengths, bad checksums and truncated streams. Some corruptions, including altered unchecked header fields, can be undetectable.

# VTS1 finite streaming envelope

`VTS1` magic, u32 version 1, u32 frame rank, zero u32 reserved, u64 frame count, u64 chunk count, and frame-rank u64 dimensions are followed by pairs of u64 byte length and standalone VTLC container bytes. Each chunk has axis zero, matching frame shape, and its own first keyframe. The stream decoder borrows input bytes; they must remain valid through decoding. `take_chunk()` can emit standalone VTLC tensors continuously; the application retains their order. `finish()` builds one VTS1 envelope and retains completed chunks until then; `finish_chunks()` returns the remainder of a drained stream.

# Field reference and parsing order

All offsets below are byte offsets relative to the beginning of the respective record. Integers are little-endian. Lengths refer to bytes unless explicitly stated otherwise.

## VTLC tensor header

| Offset | Type | Field | Constraint |
|---:|---|---|---|
| 0 | 4 bytes | Magic | ASCII `VTLC`. |
| 4 | u16 | Version | `1`. |
| 6 | u8 | Dtype | `1`, FP32 represented as 32-bit words. |
| 7 | u8 | Byte order | `1`, little-endian serialized words. |
| 8 | u32 | Rank | 1–32. |
| 12 | u32 | Time axis | Less than rank. |
| 16 | u32 | Tile elements | Nonzero. |
| 20 | u32 | Keyframe interval | Nonzero. |
| 24 | u64 | Element count | Product of dimensions. |
| 32 | rank × u64 | Shape | Dimensions in original row-major order. |

The first tile begins at `32 + 8 * rank`. If the time dimension is zero or the product of the remaining dimensions is zero, the container has no tile records. Otherwise each frame has enough consecutive tiles to cover its non-temporal elements. Tile sizes are the configured maximum except for the final tile in each frame.

## Tile record

| Offset | Type | Field |
|---:|---|---|
| 0 | u8 | Predictor ID, 0–3. |
| 1 | u8 | Codec ID, 0–5. |
| 2 | u16 | Flags: `0` or `1`. |
| 4 | u32 | Decoded element count. |
| 8 | u64 | Encoded payload length. |
| 16 | optional u32 | CRC, present only when flags is `1`. |
| 16 or 20 | bytes | Exactly payload-length bytes. |

The next tile immediately follows the payload. A decoder must validate length arithmetic before reading or allocating and should apply its configured logical-size limit. The current decoder rejects trailing bytes after the final expected record. Header fields and record length/flags are not themselves covered by the CRC; CRC input is exactly the predictor byte, codec byte, four little-endian count bytes, and payload bytes.

For example, a rank-2 shape `[2, 3]` has a 48-byte tensor header. With a tile size of at least three, two tile records follow: one for each three-element frame. The first frame can only have predictor 0; predictor 1 is also eligible for the second. Their actual payload lengths and codec IDs depend on the input and selected preset.

## VTS1 stream envelope

| Offset | Type | Field |
|---:|---|---|
| 0 | 4 bytes | ASCII `VTS1`. |
| 4 | u32 | Version `1`. |
| 8 | u32 | Frame rank, 0–31. |
| 12 | u32 | Reserved, zero. |
| 16 | u64 | Total frame count. |
| 24 | u64 | Chunk count. |
| 32 | rank × u64 | Frame shape, excluding time. |
| variable | repeated | u64 chunk byte length followed by that many VTLC bytes. |

Each chunk's VTLC tensor has a leading time axis and a remaining shape equal to the frame shape. Chunk frame counts sum to the envelope's frame count. A zero-frame envelope has zero chunks. Envelope construction and chunk iteration must still respect the borrowing and memory limitations in [STREAMING.md](STREAMING.md).
