#include "vtlc/codec.hpp"
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
int main() {
  constexpr size_t n = 262144;
  constexpr int repeats = 300;
  std::mt19937 rng(9);
  std::vector<uint32_t> x(n), a(n), b(n), out(n), reference(n);
  for (size_t i = 0; i < n; i++) {
    x[i] = rng();
    a[i] = rng();
    b[i] = rng();
  }
  std::cout << "predictor,backend,input_MiB_s,checksum\n";
  for (int p = 0; p < 4; p++) {
    vtlc::predict_xor_words(x, a, b, reference, p, vtlc::Backend::Scalar);
    for (auto be : {vtlc::Backend::Scalar, vtlc::Backend::AVX2,
                    vtlc::Backend::AVX2_FMA, vtlc::Backend::AVX512F})
      if (vtlc::available(be)) {
        auto t = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; i++)
          vtlc::predict_xor_words(x, a, b, out, p, be);
        auto e = std::chrono::steady_clock::now();
        if (out != reference)
          return 1;
        uint32_t sum = 0;
        for (auto v : out)
          sum ^= v;
        double seconds = std::chrono::duration<double>(e - t).count();
        std::cout << p << ',' << vtlc::backend_name(be) << ','
                  << (double(n) * 4 * repeats / 1048576) / seconds << ',' << sum
                  << '\n';
      }
  }
}
