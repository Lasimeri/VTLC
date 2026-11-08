#include "vtlc/codec.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#include <zstd.h>
using namespace vtlc;
using Clock = std::chrono::steady_clock;
static std::vector<uint8_t> zencode(const void *p, size_t n, int level) {
  std::vector<uint8_t> out(ZSTD_compressBound(n));
  size_t got = ZSTD_compress(out.data(), out.size(), p, n, level);
  if (ZSTD_isError(got))
    throw std::runtime_error("zstd failure");
  out.resize(got);
  return out;
}
static std::vector<uint8_t> zdecode(const std::vector<uint8_t> &in, size_t n) {
  std::vector<uint8_t> out(n);
  size_t got = ZSTD_decompress(out.data(), n, in.data(), in.size());
  if (ZSTD_isError(got) || got != n)
    throw std::runtime_error("zstd decode failure");
  return out;
}
int main() {
  constexpr size_t T = 16, N = 16384;
  std::vector<uint64_t> shape{T, N};
  std::mt19937 rng(42);
  std::cout << "dataset,codec,bytes,ratio,reduction,factor,encode_MiB_s,decode_"
               "MiB_s,encode_ms,decode_ms,process_peak_MiB\n";
  for (int kind = 0; kind < 8; kind++) {
    std::vector<float> x(T * N);
    for (size_t t = 0; t < T; t++)
      for (size_t i = 0; i < N; i++) {
        float v = 0;
        uint32_t bits = 0;
        switch (kind) {
        case 0:
          v = float(i % 100) / 100;
          break;
        case 1:
          v = std::sin(float(i) * .01f) + float(t) * .001f;
          break;
        case 2:
          v = std::sin(float(i) * .01f) + float(t) * .001f +
              float(int(rng() % 2001) - 1000) * .00001f;
          break;
        case 3:
          v = (rng() % 10) ? 0.0f : float(int(rng() % 200) - 100) / 20;
          break;
        case 4:
          v = float(int(i % 1000) - 500) * .0001f + float(t) * .000001f;
          break;
        case 5:
          v = float(rng()) / float(UINT32_MAX);
          break;
        case 6:
          bits = rng();
          std::memcpy(&v, &bits, 4);
          break;
        case 7:
          bits = (rng() % 3 == 0) ? 0x7f800001u + (rng() % 256) : rng();
          std::memcpy(&v, &bits, 4);
          break;
        }
        std::memcpy(&x[t * N + i], &v, 4);
      }
    const char *names[] = {"repetition",  "smooth",     "noisy",
                           "sparse",      "weights",    "random_float",
                           "random_bits", "exceptional"};
    const char *codecs[] = {
        "vtlc_auto",        "vtlc_scalar",       "vtlc_fast", "vtlc_fma",
        "vtlc_p1",          "vtlc_p2",           "zstd1",     "zstd3",
        "bitshuffle_zstd1", "temporal_xor_zstd1"};
    for (int codec = 0; codec < 10; codec++) {
      std::vector<double> enc, dec, peaks;
      size_t sz = 0;
      for (int r = 0; r < 3; r++) {
        struct Measurement {
          double encode, decode;
          uint64_t bytes;
        } measurement{};
        int fd[2];
        if (pipe(fd) != 0)
          throw std::runtime_error("pipe failed");
        pid_t child = fork();
        if (child < 0)
          throw std::runtime_error("fork failed");
        if (child == 0) {
          close(fd[0]);
          try {
            auto start = Clock::now();
            std::vector<uint8_t> c;
            Options opt;
            if (codec == 1)
              opt.backend = Backend::Scalar;
            if (codec == 2)
              opt.preset = Preset::Fast;
            if (codec == 3)
              opt.predictor = 3;
            if (codec == 4)
              opt.predictor = 1;
            if (codec == 5)
              opt.predictor = 2;
            if (codec < 6)
              c = compress(x, shape, opt);
            else if (codec < 8)
              c = zencode(x.data(), x.size() * 4, codec == 6 ? 1 : 3);
            else {
              std::vector<uint32_t> words(x.size());
              std::memcpy(words.data(), x.data(), x.size() * 4);
              if (codec == 8) {
                auto shuffled = bitshuffle_words(words);
                c = zencode(shuffled.data(), shuffled.size(), 1);
              } else {
                for (size_t t = T - 1; t > 0; t--)
                  for (size_t i = 0; i < N; i++)
                    words[t * N + i] ^= words[(t - 1) * N + i];
                c = zencode(words.data(), words.size() * 4, 1);
              }
            }
            auto mid = Clock::now();
            std::vector<uint32_t> result;
            if (codec < 6)
              result = decompress(c).bits;
            else if (codec < 8) {
              auto raw = zdecode(c, x.size() * 4);
              result.resize(x.size());
              std::memcpy(result.data(), raw.data(), raw.size());
            } else if (codec == 8) {
              auto raw = zdecode(c, x.size() * 4);
              result = bitunshuffle_words(raw, x.size());
            } else {
              auto raw = zdecode(c, x.size() * 4);
              result.resize(x.size());
              std::memcpy(result.data(), raw.data(), raw.size());
              for (size_t t = 1; t < T; t++)
                for (size_t i = 0; i < N; i++)
                  result[t * N + i] ^= result[(t - 1) * N + i];
            }
            auto end = Clock::now();
            if (std::memcmp(result.data(), x.data(), x.size() * 4))
              throw std::runtime_error("round trip mismatch");
            measurement.encode =
                std::chrono::duration<double, std::milli>(mid - start).count();
            measurement.decode =
                std::chrono::duration<double, std::milli>(end - mid).count();
            measurement.bytes = c.size();
            if (write(fd[1], &measurement, sizeof(measurement)) !=
                sizeof(measurement))
              _exit(2);
            _exit(0);
          } catch (...) {
            _exit(3);
          }
        }
        close(fd[1]);
        ssize_t received = read(fd[0], &measurement, sizeof(measurement));
        close(fd[0]);
        int status = 0;
        struct rusage usage{};
        if (wait4(child, &status, 0, &usage) != child || !WIFEXITED(status) ||
            WEXITSTATUS(status) || received != sizeof(measurement))
          throw std::runtime_error("benchmark child failed");
        enc.push_back(measurement.encode);
        dec.push_back(measurement.decode);
        peaks.push_back(usage.ru_maxrss / 1024.0);
        sz = measurement.bytes;
      }
      std::sort(enc.begin(), enc.end());
      std::sort(dec.begin(), dec.end());
      std::sort(peaks.begin(), peaks.end());
      double mib = x.size() * 4.0 / 1048576,
             ratio = double(sz) / (x.size() * 4);
      std::cout << names[kind] << ',' << codecs[codec] << ',' << sz << ','
                << std::fixed << std::setprecision(3) << ratio << ','
                << 1 - ratio << ',' << 1 / ratio << ',' << mib * 1000 / enc[1]
                << ',' << mib * 1000 / dec[1] << ',' << enc[1] << ',' << dec[1]
                << ',' << peaks[1] << '\n';
    }
  }
}
