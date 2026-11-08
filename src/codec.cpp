#include "vtlc/codec.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#ifdef VTLC_HAVE_ZSTD
#include <zstd.h>
#endif
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
namespace vtlc {
using B = std::vector<uint8_t>;
template <class T> struct AlignedAllocator {
  using value_type = T;
  AlignedAllocator() = default;
  template <class U> AlignedAllocator(const AlignedAllocator<U> &) noexcept {}
  T *allocate(size_t n) {
    if (n > SIZE_MAX / sizeof(T))
      throw std::bad_array_new_length();
    return static_cast<T *>(
        ::operator new(n * sizeof(T), std::align_val_t{64}));
  }
  void deallocate(T *p, size_t) noexcept {
    ::operator delete(p, std::align_val_t{64});
  }
  template <class U>
  bool operator==(const AlignedAllocator<U> &) const noexcept {
    return true;
  }
};
using AlignedWords = std::vector<uint32_t, AlignedAllocator<uint32_t>>;
static void put(B &b, uint64_t x, unsigned n) {
  for (unsigned i = 0; i < n; i++)
    b.push_back(uint8_t(x >> (8 * i)));
}
static uint32_t crc_tile(unsigned pred, unsigned codec, uint32_t count,
                         std::span<const uint8_t> payload) {
  static constexpr auto table = [] {
    std::array<uint32_t, 256> t{};
    for (uint32_t i = 0; i < 256; i++) {
      uint32_t x = i;
      for (int k = 0; k < 8; k++)
        x = (x >> 1) ^ (0xedb88320u & uint32_t(0u - (x & 1)));
      t[i] = x;
    }
    return t;
  }();
  uint32_t crc = 0xffffffffu;
  auto byte = [&](uint8_t x) { crc = (crc >> 8) ^ table[(crc ^ x) & 255]; };
  byte(uint8_t(pred));
  byte(uint8_t(codec));
  for (unsigned i = 0; i < 4; i++)
    byte(uint8_t(count >> (8 * i)));
  for (auto b : payload)
    byte(b);
  return ~crc;
}
struct Reader {
  std::span<const uint8_t> b;
  size_t p = 0;
  uint64_t get(unsigned n) {
    if (n > b.size() - p)
      throw std::runtime_error("truncated container");
    uint64_t x = 0;
    for (unsigned i = 0; i < n; i++)
      x |= uint64_t(b[p++]) << (8 * i);
    return x;
  }
  std::span<const uint8_t> take(size_t n) {
    if (n > b.size() - p)
      throw std::runtime_error("truncated payload");
    auto s = b.subspan(p, n);
    p += n;
    return s;
  }
};
static uint64_t mul(uint64_t a, uint64_t b) {
  if (b && a > UINT64_MAX / b)
    throw std::runtime_error("shape overflow");
  return a * b;
}
bool available(Backend b) {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_cpu_init();
  switch (b) {
  case Backend::Auto:
  case Backend::Scalar:
    return true;
  case Backend::AVX2:
    return __builtin_cpu_supports("avx2");
  case Backend::AVX2_FMA:
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  case Backend::AVX512F:
    return __builtin_cpu_supports("avx512f");
  }
#else
  return b == Backend::Auto || b == Backend::Scalar;
#endif
  return false;
}
const char *backend_name(Backend b) {
  switch (b) {
  case Backend::Auto:
    return "auto";
  case Backend::Scalar:
    return "scalar";
  case Backend::AVX2:
    return "avx2";
  case Backend::AVX2_FMA:
    return "avx2-fma";
  case Backend::AVX512F:
    return "avx512f";
  }
  return "unknown";
}
static Backend choose(Backend b) {
  if (!available(b))
    throw std::runtime_error("backend unavailable");
  if (b != Backend::Auto)
    return b;
  if (available(Backend::AVX512F))
    return Backend::AVX512F;
  if (available(Backend::AVX2_FMA))
    return Backend::AVX2_FMA;
  if (available(Backend::AVX2))
    return Backend::AVX2;
  return Backend::Scalar;
}
// Integer prediction and explicitly fused floating point prediction; XOR is its
// own inverse.
static bool normal(uint32_t x) {
  auto e = x & 0x7f800000u;
  return e != 0 && e != 0x7f800000u;
}
static uint32_t fp_predict(uint32_t a, uint32_t b) {
  if (!normal(a) || !normal(b))
    return a;
  float x = std::bit_cast<float>(a), y = std::bit_cast<float>(b);
  uint32_t q = std::bit_cast<uint32_t>(std::fma(2.0f, x, -y));
  return normal(q) ? q : a;
}
static void scalar(const uint32_t *x, const uint32_t *a, const uint32_t *b,
                   uint32_t *out, size_t n, int pred) {
  for (size_t i = 0; i < n; i++) {
    uint32_t p = pred == 0   ? 0
                 : pred == 1 ? a[i]
                 : pred == 2 ? a[i] + a[i] - b[i]
                             : fp_predict(a[i], b[i]);
    out[i] = x[i] ^ p;
  }
}
#if defined(__x86_64__) || defined(__i386__)
// Replace exceptional operands before evaluation so signaling NaNs never reach
// FMA.
__attribute__((target("avx2,fma"))) static void
fma256(const uint32_t *x, const uint32_t *a, const uint32_t *b, uint32_t *out,
       size_t n) {
  size_t i = 0;
  const auto exp = _mm256_set1_epi32(0x7f800000), zero = _mm256_setzero_si256(),
             sign = _mm256_set1_epi32(int(0x80000000u));
  for (; i + 8 <= n; i += 8) {
    auto aa = _mm256_loadu_si256((const __m256i *)(a + i)),
         bb = _mm256_loadu_si256((const __m256i *)(b + i));
    auto ea = _mm256_and_si256(aa, exp), eb = _mm256_and_si256(bb, exp);
    auto bad = _mm256_or_si256(_mm256_or_si256(_mm256_cmpeq_epi32(ea, zero),
                                               _mm256_cmpeq_epi32(ea, exp)),
                               _mm256_or_si256(_mm256_cmpeq_epi32(eb, zero),
                                               _mm256_cmpeq_epi32(eb, exp)));
    auto safea = _mm256_blendv_epi8(aa, _mm256_set1_epi32(0x3f800000), bad);
    auto safeb = _mm256_blendv_epi8(bb, _mm256_set1_epi32(0x3f800000), bad);
    auto q = _mm256_castps_si256(
        _mm256_fmadd_ps(_mm256_set1_ps(2.f), _mm256_castsi256_ps(safea),
                        _mm256_castsi256_ps(_mm256_xor_si256(safeb, sign))));
    auto eq = _mm256_and_si256(q, exp);
    bad = _mm256_or_si256(bad, _mm256_or_si256(_mm256_cmpeq_epi32(eq, zero),
                                               _mm256_cmpeq_epi32(eq, exp)));
    q = _mm256_blendv_epi8(q, aa, bad);
    _mm256_storeu_si256(
        (__m256i *)(out + i),
        _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(x + i)), q));
  }
  scalar(x + i, a + i, b + i, out + i, n - i, 3);
}
__attribute__((target("avx512f"))) static void fma512(const uint32_t *x,
                                                      const uint32_t *a,
                                                      const uint32_t *b,
                                                      uint32_t *out, size_t n) {
  size_t i = 0;
  const auto exp = _mm512_set1_epi32(0x7f800000), zero = _mm512_setzero_si512(),
             sign = _mm512_set1_epi32(int(0x80000000u)),
             one = _mm512_set1_epi32(0x3f800000);
  for (; i + 16 <= n; i += 16) {
    auto aa = _mm512_loadu_si512(a + i), bb = _mm512_loadu_si512(b + i);
    auto ea = _mm512_and_si512(aa, exp), eb = _mm512_and_si512(bb, exp);
    auto bad =
        _mm512_cmpeq_epi32_mask(ea, zero) | _mm512_cmpeq_epi32_mask(ea, exp) |
        _mm512_cmpeq_epi32_mask(eb, zero) | _mm512_cmpeq_epi32_mask(eb, exp);
    auto safea = _mm512_mask_mov_epi32(aa, bad, one),
         safeb = _mm512_mask_mov_epi32(bb, bad, one);
    auto q = _mm512_castps_si512(
        _mm512_fmadd_ps(_mm512_set1_ps(2.f), _mm512_castsi512_ps(safea),
                        _mm512_castsi512_ps(_mm512_xor_si512(safeb, sign))));
    auto eq = _mm512_and_si512(q, exp);
    bad |= _mm512_cmpeq_epi32_mask(eq, zero) | _mm512_cmpeq_epi32_mask(eq, exp);
    q = _mm512_mask_mov_epi32(q, bad, aa);
    _mm512_storeu_si512(out + i,
                        _mm512_xor_si512(_mm512_loadu_si512(x + i), q));
  }
  scalar(x + i, a + i, b + i, out + i, n - i, 3);
}
__attribute__((target("avx2"))) static void
avx2(const uint32_t *x, const uint32_t *a, const uint32_t *b, uint32_t *out,
     size_t n, int p) {
  size_t i = 0;
  const bool aligned =
      ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(a) |
        reinterpret_cast<uintptr_t>(b) | reinterpret_cast<uintptr_t>(out)) &
       31) == 0;
  for (; i + 8 <= n; i += 8) {
    auto v = aligned ? _mm256_load_si256((const __m256i *)(x + i))
                     : _mm256_loadu_si256((const __m256i *)(x + i));
    auto q = _mm256_setzero_si256();
    if (p) {
      q = aligned ? _mm256_load_si256((const __m256i *)(a + i))
                  : _mm256_loadu_si256((const __m256i *)(a + i));
      if (p == 2)
        q = _mm256_sub_epi32(
            _mm256_add_epi32(q, q),
            aligned ? _mm256_load_si256((const __m256i *)(b + i))
                    : _mm256_loadu_si256((const __m256i *)(b + i)));
    }
    if (aligned)
      _mm256_store_si256((__m256i *)(out + i), _mm256_xor_si256(v, q));
    else
      _mm256_storeu_si256((__m256i *)(out + i), _mm256_xor_si256(v, q));
  }
  scalar(x + i, a ? a + i : nullptr, b ? b + i : nullptr, out + i, n - i, p);
}
__attribute__((target("avx512f"))) static void
avx512(const uint32_t *x, const uint32_t *a, const uint32_t *b, uint32_t *out,
       size_t n, int p) {
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    auto v = _mm512_loadu_si512(x + i);
    auto q = _mm512_setzero_si512();
    if (p) {
      q = _mm512_loadu_si512(a + i);
      if (p == 2)
        q = _mm512_sub_epi32(_mm512_add_epi32(q, q), _mm512_loadu_si512(b + i));
    }
    _mm512_storeu_si512(out + i, _mm512_xor_si512(v, q));
  }
  scalar(x + i, a ? a + i : nullptr, b ? b + i : nullptr, out + i, n - i, p);
}
#endif
static void transform(const uint32_t *x, const uint32_t *a, const uint32_t *b,
                      uint32_t *out, size_t n, int p, Backend be) {
#if defined(__x86_64__) || defined(__i386__)
  if (p == 3) {
    unsigned saved = _mm_getcsr();
    _mm_setcsr((saved & ~unsigned(0xe07f)) | 0x1f80);
    if (be == Backend::AVX512F)
      fma512(x, a, b, out, n);
    else if (be == Backend::AVX2_FMA)
      fma256(x, a, b, out, n);
    else
      scalar(x, a, b, out, n, p);
    _mm_setcsr(saved);
    return;
  }
  if (be == Backend::AVX512F) {
    avx512(x, a, b, out, n, p);
    return;
  }
  if (be == Backend::AVX2 || be == Backend::AVX2_FMA) {
    avx2(x, a, b, out, n, p);
    return;
  }
#else
  if (p == 3)
    throw std::runtime_error(
        "FMA predictor semantics unsupported on this platform");
#endif
  scalar(x, a, b, out, n, p);
}
void predict_xor_words(std::span<const uint32_t> x, std::span<const uint32_t> a,
                       std::span<const uint32_t> b, std::span<uint32_t> out,
                       int p, Backend be) {
  if (p < 0 || p > 3 || out.size() != x.size() ||
      (p >= 1 && a.size() != x.size()) || (p >= 2 && b.size() != x.size()))
    throw std::runtime_error("invalid prediction arguments");
  if (x.empty()) {
    choose(be);
    return;
  }
  transform(x.data(), a.data(), b.data(), out.data(), x.size(), p, choose(be));
}
static uint64_t index(uint64_t t, uint64_t i, uint64_t inner, uint64_t times) {
  return (i / inner) * times * inner + t * inner + i % inner;
}
static Metadata header(Reader &r, uint64_t limit) {
  if (r.get(4) != 0x434c5456 || r.get(2) != 1 || r.get(1) != 1 || r.get(1) != 1)
    throw std::runtime_error("invalid magic, version, dtype or byte order");
  uint64_t dims = r.get(4);
  Metadata m{};
  m.time_axis = uint32_t(r.get(4));
  m.tile_elements = uint32_t(r.get(4));
  m.keyframe_interval = uint32_t(r.get(4));
  m.elements = r.get(8);
  if (!dims || dims > 32 || m.time_axis >= dims || !m.tile_elements ||
      !m.keyframe_interval)
    throw std::runtime_error("invalid header");
  uint64_t count = 1;
  for (size_t i = 0; i < dims; i++) {
    auto d = r.get(8);
    m.shape.push_back(d);
    count = mul(count, d);
  }
  if (count != m.elements || count > limit / 4 || count > SIZE_MAX / 4)
    throw std::runtime_error("invalid size or memory limit exceeded");
  return m;
}
Metadata inspect(std::span<const uint8_t> x, uint64_t l) {
  Reader r{x};
  return header(r, l);
}
static B rawbytes(const std::vector<uint32_t> &v) {
  if constexpr (std::endian::native == std::endian::little) {
    B o(v.size() * 4);
    if (!o.empty())
      std::memcpy(o.data(), v.data(), o.size());
    return o;
  }
  B o;
  o.reserve(v.size() * 4);
  for (auto x : v)
    put(o, x, 4);
  return o;
}
// Eight input words become 32 bytes, each containing one bit from every word.
static B bitshuffle_scalar(const std::vector<uint32_t> &v) {
  B out;
  out.reserve(v.size() * 4);
  size_t i = 0;
  for (; i + 8 <= v.size(); i += 8) {
    for (unsigned bit = 0; bit < 32; bit++) {
      uint8_t plane = 0;
      for (unsigned k = 0; k < 8; k++)
        plane |= uint8_t(((v[i + k] >> bit) & 1u) << k);
      out.push_back(plane);
    }
  }
  for (; i < v.size(); i++)
    put(out, v[i], 4);
  return out;
}
#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2"))) static B
bitshuffle_avx2(const std::vector<uint32_t> &v) {
  B out;
  out.reserve(v.size() * 4);
  size_t i = 0;
  for (; i + 8 <= v.size(); i += 8) {
    auto x = _mm256_loadu_si256((const __m256i *)(v.data() + i));
    for (unsigned bit = 0; bit < 32; bit++) {
      auto shift = _mm256_set1_epi32(int(31 - bit));
      auto y = _mm256_sllv_epi32(x, shift);
      out.push_back(uint8_t(_mm256_movemask_ps(_mm256_castsi256_ps(y))));
    }
  }
  for (; i < v.size(); i++)
    put(out, v[i], 4);
  return out;
}
#endif
static B bitshuffle(const std::vector<uint32_t> &v, Backend be) {
#if defined(__x86_64__) || defined(__i386__)
  if (be != Backend::Scalar && available(Backend::AVX2))
    return bitshuffle_avx2(v);
#else
  (void)be;
#endif
  return bitshuffle_scalar(v);
}
static std::vector<uint32_t> unshuffle_scalar(std::span<const uint8_t> data,
                                              size_t n) {
  std::vector<uint32_t> out(n);
  Reader r{data};
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    for (unsigned bit = 0; bit < 32; bit++) {
      auto plane = r.get(1);
      for (unsigned k = 0; k < 8; k++)
        out[i + k] |= uint32_t((plane >> k) & 1) << bit;
    }
  }
  for (; i < n; i++)
    out[i] = uint32_t(r.get(4));
  return out;
}
#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2"))) static std::vector<uint32_t>
unshuffle_avx2(std::span<const uint8_t> data, size_t n) {
  std::vector<uint32_t> out(n);
  size_t pos = 0, i = 0;
  const auto lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
             one = _mm256_set1_epi32(1);
  for (; i + 8 <= n; i += 8) {
    auto words = _mm256_setzero_si256();
    for (unsigned bit = 0; bit < 32; bit++) {
      auto plane = _mm256_set1_epi32(data[pos++]);
      auto bits = _mm256_and_si256(_mm256_srlv_epi32(plane, lanes), one);
      words = _mm256_or_si256(
          words, _mm256_sllv_epi32(bits, _mm256_set1_epi32(int(bit))));
    }
    _mm256_storeu_si256((__m256i *)(out.data() + i), words);
  }
  Reader r{data.subspan(pos)};
  for (; i < n; i++)
    out[i] = uint32_t(r.get(4));
  return out;
}
#endif
static std::vector<uint32_t> unshuffle(std::span<const uint8_t> data, size_t n,
                                       Backend be) {
  if (n > SIZE_MAX / 4 || data.size() != n * 4)
    throw std::runtime_error("invalid bitshuffle size");
#if defined(__x86_64__) || defined(__i386__)
  if (be != Backend::Scalar && available(Backend::AVX2))
    return unshuffle_avx2(data, n);
#else
  (void)be;
#endif
  return unshuffle_scalar(data, n);
}
std::vector<uint8_t> bitshuffle_words(std::span<const uint32_t> input,
                                      Backend backend) {
  return bitshuffle(std::vector<uint32_t>(input.begin(), input.end()),
                    choose(backend));
}
std::vector<uint32_t> bitunshuffle_words(std::span<const uint8_t> input,
                                         size_t n, Backend be) {
  return unshuffle(input, n, choose(be));
}
// Packed blocks use a single exact width per group of 128, with a raw fallback
// at tile level.
#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2"))) static uint32_t block_or_avx2(const uint32_t *p,
                                                              size_t n) {
  auto accum = _mm256_setzero_si256();
  size_t i = 0;
  for (; i + 8 <= n; i += 8)
    accum =
        _mm256_or_si256(accum, _mm256_loadu_si256((const __m256i *)(p + i)));
  alignas(32) uint32_t lanes[8];
  _mm256_store_si256((__m256i *)lanes, accum);
  uint32_t result = 0;
  for (auto x : lanes)
    result |= x;
  for (; i < n; i++)
    result |= p[i];
  return result;
}
__attribute__((target("avx2"))) static size_t
pack_bytes_avx2(const uint32_t *p, size_t n, unsigned width, B &out) {
  size_t i = 0;
  const auto m8 = _mm256_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1,
                                   -1, -1, -1, -1, 0, 4, 8, 12, -1, -1, -1, -1,
                                   -1, -1, -1, -1, -1, -1, -1, -1);
  const auto m16 = _mm256_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1,
                                    -1, -1, -1, -1, 0, 1, 4, 5, 8, 9, 12, 13,
                                    -1, -1, -1, -1, -1, -1, -1, -1);
  for (; i + 8 <= n; i += 8) {
    auto x = _mm256_loadu_si256((const __m256i *)(p + i));
    auto shuffled = _mm256_shuffle_epi8(x, width == 8 ? m8 : m16);
    auto lo = _mm256_castsi256_si128(shuffled),
         hi = _mm256_extracti128_si256(shuffled, 1);
    if (width == 8) {
      put(out, uint32_t(_mm_cvtsi128_si32(lo)), 4);
      put(out, uint32_t(_mm_cvtsi128_si32(hi)), 4);
    } else {
      put(out, uint64_t(_mm_cvtsi128_si64(lo)), 8);
      put(out, uint64_t(_mm_cvtsi128_si64(hi)), 8);
    }
  }
  return i;
}
__attribute__((target("avx2"))) static size_t
unpack_bytes_avx2(const uint8_t *p, size_t n, unsigned width, uint32_t *out) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256i x;
    if (width == 8) {
      auto bytes = _mm_loadl_epi64((const __m128i *)(p + i));
      x = _mm256_cvtepu8_epi32(bytes);
    } else {
      auto bytes = _mm_loadu_si128((const __m128i *)(p + i * 2));
      x = _mm256_cvtepu16_epi32(bytes);
    }
    _mm256_storeu_si256((__m256i *)(out + i), x);
  }
  return i;
}
#endif
static B pack(const std::vector<uint32_t> &v, Backend backend) {
  B o;
#if defined(__x86_64__) || defined(__i386__)
  const bool vectorized =
      backend != Backend::Scalar && available(Backend::AVX2);
#else
  (void)backend;
#endif
  for (size_t j = 0; j < v.size(); j += 128) {
    size_t n = std::min(size_t(128), v.size() - j);
    uint32_t all = 0;
#if defined(__x86_64__) || defined(__i386__)
    if (vectorized)
      all = block_or_avx2(v.data() + j, n);
    else
#endif
    {
      for (size_t i = 0; i < n; i++)
        all |= v[j + i];
    }
    unsigned w = std::bit_width(all);
    o.push_back(uint8_t(w));
    if (w == 8 || w == 16) {
      size_t i = 0;
#if defined(__x86_64__) || defined(__i386__)
      if (vectorized)
        i = pack_bytes_avx2(v.data() + j, n, w, o);
#endif
      for (; i < n; i++) {
        put(o, v[j + i], w / 8);
      }
      continue;
    }
    uint64_t acc = 0;
    unsigned bits = 0;
    for (size_t i = 0; i < n; i++) {
      acc |= uint64_t(v[j + i]) << bits;
      bits += w;
      while (bits >= 8) {
        o.push_back(uint8_t(acc));
        acc >>= 8;
        bits -= 8;
      }
    }
    if (bits)
      o.push_back(uint8_t(acc));
  }
  return o;
}
static std::vector<uint32_t> unpack(std::span<const uint8_t> data, size_t n,
                                    Backend backend) {
  Reader r{data};
  std::vector<uint32_t> o(n);
#if defined(__x86_64__) || defined(__i386__)
  const bool vectorized =
      backend != Backend::Scalar && available(Backend::AVX2);
#else
  (void)backend;
#endif
  for (size_t j = 0; j < n; j += 128) {
    size_t count = std::min(size_t(128), n - j);
    unsigned w = r.get(1);
    if (w > 32)
      throw std::runtime_error("invalid bit width");
    if (w == 8 || w == 16) {
      auto bytes = r.take(count * (w / 8));
      size_t i = 0;
#if defined(__x86_64__) || defined(__i386__)
      if (vectorized)
        i = unpack_bytes_avx2(bytes.data(), count, w, o.data() + j);
#endif
      Reader r2{bytes};
      r2.p = i * (w / 8);
      for (; i < count; i++)
        o[j + i] = uint32_t(r2.get(w / 8));
      continue;
    }
    uint64_t acc = 0;
    unsigned bits = 0;
    for (size_t i = 0; i < count; i++) {
      while (bits < w) {
        acc |= r.get(1) << bits;
        bits += 8;
      }
      o[j + i] =
          w == 32 ? uint32_t(acc) : uint32_t(acc & ((uint64_t(1) << w) - 1));
      acc >>= w;
      bits -= w;
    }
    if (acc)
      throw std::runtime_error("invalid padding");
  }
  if (r.p != data.size())
    throw std::runtime_error("trailing packed bytes");
  return o;
}
static void varint(B &out, uint64_t v) {
  while (v >= 128) {
    out.push_back(uint8_t(v) | 0x80);
    v >>= 7;
  }
  out.push_back(uint8_t(v));
}
static uint64_t readvar(Reader &r) {
  uint64_t v = 0;
  for (unsigned i = 0; i < 10; i++) {
    uint8_t b = r.get(1);
    if (i == 9 && (b & 0xfe))
      throw std::runtime_error("varint overflow");
    v |= uint64_t(b & 127) << (i * 7);
    if (!(b & 128))
      return v;
  }
  throw std::runtime_error("invalid varint");
}
static size_t scalar_run(const uint32_t *p, size_t n, bool nonzero) {
  size_t i = 0;
  while (i < n && (p[i] != 0) == nonzero)
    i++;
  return i;
}
#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2"))) static size_t
vector_run(const uint32_t *p, size_t n, bool nonzero) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    auto v = _mm256_loadu_si256((const __m256i *)(p + i));
    unsigned zeros = unsigned(_mm256_movemask_ps(
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(v, _mm256_setzero_si256()))));
    unsigned mismatch = nonzero ? zeros : (~zeros & 255u);
    if (mismatch)
      return i + std::countr_zero(mismatch);
  }
  return i + scalar_run(p + i, n - i, nonzero);
}
#endif
static B rle(const std::vector<uint32_t> &v, Backend backend) {
  B out;
#if defined(__x86_64__) || defined(__i386__)
  const bool vectorized =
      backend != Backend::Scalar && available(Backend::AVX2);
#else
  (void)backend;
#endif
  for (size_t i = 0; i < v.size();) {
    bool nonzero = v[i] != 0;
    size_t count;
#if defined(__x86_64__) || defined(__i386__)
    if (vectorized) {
      count = vector_run(v.data() + i, v.size() - i, nonzero);
    } else {
      count = scalar_run(v.data() + i, v.size() - i, nonzero);
    }
#else
    count = scalar_run(v.data() + i, v.size() - i, nonzero);
#endif
    size_t j = i + count;
    varint(out, (uint64_t(count) << 1) | nonzero);
    if (nonzero)
      for (size_t k = i; k < j; k++)
        put(out, v[k], 4);
    i = j;
  }
  return out;
}
static std::vector<uint32_t> unrle(std::span<const uint8_t> data, size_t n) {
  Reader r{data};
  std::vector<uint32_t> out(n);
  size_t pos = 0;
  while (pos < n) {
    auto token = readvar(r), count = token >> 1;
    if (!count || count > n - pos)
      throw std::runtime_error("invalid run length");
    if (token & 1)
      for (size_t i = 0; i < count; i++)
        out[pos + i] = uint32_t(r.get(4));
    pos += count;
  }
  if (r.p != data.size())
    throw std::runtime_error("trailing run bytes");
  return out;
}
struct BitWriter {
  B out;
  uint64_t acc = 0;
  unsigned used = 0;
  void write(uint32_t value, unsigned n) {
    acc |= uint64_t(value) << used;
    used += n;
    while (used >= 8) {
      out.push_back(uint8_t(acc));
      acc >>= 8;
      used -= 8;
    }
  }
  B finish() {
    if (used)
      out.push_back(uint8_t(acc));
    return std::move(out);
  }
};
struct BitReader {
  Reader r;
  uint64_t acc = 0;
  unsigned used = 0;
  uint32_t read(unsigned n) {
    while (used < n) {
      acc |= r.get(1) << used;
      used += 8;
    }
    uint32_t x =
        n == 32 ? uint32_t(acc) : uint32_t(acc & ((uint64_t(1) << n) - 1));
    acc >>= n;
    used -= n;
    return x;
  }
  void finish() {
    if (acc || r.p != r.b.size())
      throw std::runtime_error("invalid bitstream padding");
  }
};
static B gorilla(const std::vector<uint32_t> &v) {
  BitWriter w;
  unsigned oldlead = 0, oldtrail = 0;
  bool window = false;
  for (auto x : v) {
    if (!x) {
      w.write(0, 1);
      continue;
    }
    w.write(1, 1);
    unsigned lead = std::countl_zero(x), trail = std::countr_zero(x);
    if (window && lead >= oldlead && trail >= oldtrail &&
        (32 - oldlead - oldtrail) <= (32 - lead - trail) + 10) {
      w.write(0, 1);
      w.write(x >> oldtrail, 32 - oldlead - oldtrail);
    } else {
      w.write(1, 1);
      w.write(lead, 5);
      unsigned width = 32 - lead - trail;
      w.write(width - 1, 5);
      w.write(x >> trail, width);
      oldlead = lead;
      oldtrail = trail;
      window = true;
    }
  }
  return w.finish();
}
static std::vector<uint32_t> ungorilla(std::span<const uint8_t> data,
                                       size_t n) {
  BitReader r{{data}};
  std::vector<uint32_t> out(n);
  unsigned lead = 0, trail = 0;
  bool window = false;
  for (auto &x : out) {
    if (!r.read(1))
      continue;
    if (r.read(1)) {
      lead = r.read(5);
      unsigned width = r.read(5) + 1;
      if (lead + width > 32)
        throw std::runtime_error("invalid significant window");
      trail = 32 - lead - width;
      window = true;
    } else if (!window)
      throw std::runtime_error("missing significant window");
    unsigned width = 32 - lead - trail;
    x = r.read(width) << trail;
    if (!x)
      throw std::runtime_error("invalid zero residual");
  }
  r.finish();
  return out;
}
static void zstd_decode(B &raw, std::span<const uint8_t> data) {
#ifdef VTLC_HAVE_ZSTD
  size_t got =
      ZSTD_decompress(raw.data(), raw.size(), data.data(), data.size());
  if (ZSTD_isError(got) || got != raw.size())
    throw std::runtime_error("invalid zstd payload");
#else
  (void)raw;
  (void)data;
  throw std::runtime_error("Zstandard support is disabled");
#endif
}
std::vector<uint8_t> compress(std::span<const float> input,
                              std::span<const uint64_t> shape,
                              const Options &opt) {
  if (shape.empty() || shape.size() > 32 || opt.time_axis >= shape.size() ||
      !opt.tile_elements || !opt.keyframe_interval || opt.predictor < -1 ||
      opt.predictor > 3)
    throw std::runtime_error("invalid options or shape");
#if !defined(__x86_64__) && !defined(__i386__)
  if (opt.predictor == 3)
    throw std::runtime_error(
        "FMA predictor semantics unsupported on this platform");
#endif
  uint64_t total = 1, inner = 1;
  for (auto d : shape)
    total = mul(total, d);
  for (size_t k = opt.time_axis + 1; k < shape.size(); k++)
    inner = mul(inner, shape[k]);
  if (total != input.size() || total > opt.memory_limit / 4)
    throw std::runtime_error("size mismatch or memory limit exceeded");
  auto be = choose(opt.backend);
  uint64_t times = shape[opt.time_axis], n = times ? total / times : 0;
  B out;
  put(out, 0x434c5456, 4);
  put(out, 1, 2);
  put(out, 1, 1);
  put(out, 1, 1);
  put(out, shape.size(), 4);
  put(out, opt.time_axis, 4);
  put(out, opt.tile_elements, 4);
  put(out, opt.keyframe_interval, 4);
  put(out, total, 8);
  for (auto d : shape)
    put(out, d, 8);
  AlignedWords a(n), b(n), x(n);
  std::vector<uint32_t> candidate;
  for (uint64_t t = 0; t < times && n; t++) {
    for (uint64_t i = 0; i < n; i++) {
      auto ix = index(t, i, inner, times);
      std::memcpy(&x[i], input.data() + ix, 4);
    }
    int maxp = t % opt.keyframe_interval == 0
                   ? 0
                   : (t % opt.keyframe_interval == 1 ? 1 : 3);
#if !defined(__x86_64__) && !defined(__i386__)
    maxp = std::min(maxp, 2);
#endif
    for (uint64_t pos = 0; pos < n; pos += opt.tile_elements) {
      size_t len = std::min<uint64_t>(opt.tile_elements, n - pos);
      candidate.resize(len);
      int selected = 0;
      [[maybe_unused]] uint64_t estimate = UINT64_MAX;
      if (opt.predictor >= 0)
        selected = std::min(opt.predictor, maxp);
      else if (opt.preset != Preset::Ratio) {
        uint64_t bestscore = UINT64_MAX;
        int last = opt.preset == Preset::Fast ? std::min(maxp, 2) : maxp;
        for (int p = 0; p <= last; p++) {
          transform(x.data() + pos, a.data() + pos, b.data() + pos,
                    candidate.data(), len, p, be);
          uint64_t score = 0;
          for (size_t i = 0; i < len; i += 16) {
            uint32_t v = candidate[i];
            score += v ? 1 + (std::bit_width(v) + 7) / 8 : 0;
          }
          if (score < bestscore) {
            bestscore = score;
            selected = p;
          }
        }
        estimate = bestscore;
      }
      B best;
      int bestp = 0, bestcodec = 0;
      for (int p = 0; p <= maxp; p++) {
        if ((opt.predictor >= 0 || opt.preset != Preset::Ratio) &&
            p != selected)
          continue;
        transform(x.data() + pos, a.data() + pos, b.data() + pos,
                  candidate.data(), len, p, be);
        auto consider = [&](B &&z, int c) {
          if (best.empty() || z.size() < best.size()) {
            best = std::move(z);
            bestp = p;
            bestcodec = c;
          }
        };
        B raw = rawbytes(candidate);
        consider(B(raw), 0);
        consider(pack(candidate, be), 1);
        consider(rle(candidate, be), 4);
        if (opt.preset == Preset::Ratio)
          consider(gorilla(candidate), 5);
#ifdef VTLC_HAVE_ZSTD
        if (opt.preset != Preset::Fast &&
            (opt.preset != Preset::Adaptive || estimate == UINT64_MAX ||
             estimate <= ((len + 15) / 16) * 4)) {
          B z(ZSTD_compressBound(raw.size()));
          size_t s = ZSTD_compress(z.data(), z.size(), raw.data(), raw.size(),
                                   opt.preset == Preset::Ratio ? 3 : 1);
          if (ZSTD_isError(s))
            throw std::runtime_error(ZSTD_getErrorName(s));
          z.resize(s);
          consider(std::move(z), 2);
          if (opt.preset == Preset::Ratio) {
            auto sh = bitshuffle(candidate, be);
            B zs(ZSTD_compressBound(sh.size()));
            size_t sn =
                ZSTD_compress(zs.data(), zs.size(), sh.data(), sh.size(), 3);
            if (ZSTD_isError(sn))
              throw std::runtime_error(ZSTD_getErrorName(sn));
            zs.resize(sn);
            consider(std::move(zs), 3);
          }
        }
#endif
      }
      put(out, bestp, 1);
      put(out, bestcodec, 1);
      put(out, opt.checksum ? 1 : 0, 2);
      put(out, len, 4);
      put(out, best.size(), 8);
      if (opt.checksum)
        put(out, crc_tile(bestp, bestcodec, uint32_t(len), best), 4);
      out.insert(out.end(), best.begin(), best.end());
    }
    b.swap(a);
    a.swap(x);
  }
  return out;
}

Tensor decompress(std::span<const uint8_t> input, Backend backend,
                  uint64_t limit) {
  Reader r{input};
  auto m = header(r, limit);
  auto be = choose(backend);
  uint64_t times = m.shape[m.time_axis], n = times ? m.elements / times : 0,
           inner = 1;
  for (size_t k = m.time_axis + 1; k < m.shape.size(); k++)
    inner = mul(inner, m.shape[k]);
  Tensor out{m.shape, std::vector<uint32_t>(m.elements)};
  AlignedWords a(n), b(n), x(n);
  for (uint64_t t = 0; t < times && n; t++) {
    for (uint64_t pos = 0; pos < n; pos += m.tile_elements) {
      size_t len = std::min<uint64_t>(m.tile_elements, n - pos);
      unsigned p = r.get(1), codec = r.get(1);
      unsigned flags = r.get(2);
      if (flags > 1 || r.get(4) != len)
        throw std::runtime_error("invalid tile");
      uint64_t size = r.get(8);
      uint32_t checksum = flags ? uint32_t(r.get(4)) : 0;
      if (size > r.b.size() - r.p || size > SIZE_MAX)
        throw std::runtime_error("invalid payload size");
      auto data = r.take(size);
      if (flags && checksum != crc_tile(p, codec, uint32_t(len), data))
        throw std::runtime_error("tile checksum mismatch");
      if (p > 3 ||
          p > (t % m.keyframe_interval == 0
                   ? 0
                   : (t % m.keyframe_interval == 1 ? 1 : 3)) ||
          codec > 5)
        throw std::runtime_error("invalid tile mode");
      std::vector<uint32_t> residual;
      if (codec == 1)
        residual = unpack(data, len, be);
      else if (codec == 4)
        residual = unrle(data, len);
      else if (codec == 5)
        residual = ungorilla(data, len);
      else {
        B raw;
        if (codec == 0) {
          if (size != len * 4)
            throw std::runtime_error("invalid raw size");
          residual.resize(len);
          if constexpr (std::endian::native == std::endian::little) {
            if (!data.empty())
              std::memcpy(residual.data(), data.data(), data.size());
          } else {
            Reader rr{data};
            for (auto &v : residual)
              v = uint32_t(rr.get(4));
          }
        } else {
          raw.resize(len * 4);
          zstd_decode(raw, data);
          if (codec == 3)
            residual = unshuffle(raw, len, be);
          else {
            residual.resize(len);
            if constexpr (std::endian::native == std::endian::little) {
              if (!raw.empty())
                std::memcpy(residual.data(), raw.data(), raw.size());
            } else {
              Reader rr{raw};
              for (auto &v : residual)
                v = uint32_t(rr.get(4));
            }
          }
        }
      }
      transform(residual.data(), a.data() + pos, b.data() + pos, x.data() + pos,
                len, p, be);
    }
    for (uint64_t i = 0; i < n; i++)
      out.bits[index(t, i, inner, times)] = x[i];
    b.swap(a);
    a.swap(x);
  }
  if (r.p != input.size())
    throw std::runtime_error("trailing data");
  return out;
}
} // namespace vtlc
static thread_local std::string error_text;
extern "C" const char *vtlc_error() { return error_text.c_str(); }
extern "C" void vtlc_free(void *p) { std::free(p); }
extern "C" int vtlc_compress_c(const float *p, const uint64_t *shape,
                               unsigned dims, unsigned axis,
                               unsigned char **out, uint64_t *size) {
  try {
    if (!out || !size || !shape || !dims || dims > 32)
      throw std::runtime_error("invalid C API argument");
    *out = nullptr;
    *size = 0;
    uint64_t n = 1;
    for (unsigned i = 0; i < dims; i++) {
      if (shape[i] && n > UINT64_MAX / shape[i])
        throw std::runtime_error("overflow");
      n *= shape[i];
    }
    if (n > SIZE_MAX || (!p && n))
      throw std::runtime_error("invalid input pointer or size");
    vtlc::Options o;
    o.time_axis = axis;
    auto v = vtlc::compress({p, size_t(n)}, {shape, dims}, o);
    *out = (unsigned char *)std::malloc(v.size());
    if (!*out && v.size())
      throw std::bad_alloc();
    if (!v.empty())
      std::memcpy(*out, v.data(), v.size());
    *size = v.size();
    return 0;
  } catch (const std::exception &e) {
    error_text = e.what();
    return -1;
  }
}
extern "C" int vtlc_decompress_c(const unsigned char *p, uint64_t size,
                                 unsigned char **out, uint64_t *n,
                                 uint64_t **shape, unsigned *dims) {
  try {
    if (!out || !n || !shape || !dims || (!p && size) || size > SIZE_MAX)
      throw std::runtime_error("invalid C API argument");
    *out = nullptr;
    *shape = nullptr;
    *n = 0;
    *dims = 0;
    auto t = vtlc::decompress({p, size_t(size)});
    *out = (unsigned char *)std::malloc(t.bits.size() * 4 + 1);
    *shape = (uint64_t *)std::malloc(t.shape.size() * 8);
    if (!*out || !*shape) {
      std::free(*out);
      std::free(*shape);
      *out = nullptr;
      *shape = nullptr;
      throw std::bad_alloc();
    }
    if (!t.bits.empty())
      std::memcpy(*out, t.bits.data(), t.bits.size() * 4);
    std::memcpy(*shape, t.shape.data(), t.shape.size() * 8);
    *n = t.bits.size();
    *dims = unsigned(t.shape.size());
    return 0;
  } catch (const std::exception &e) {
    error_text = e.what();
    return -1;
  }
}
