#pragma once
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace vtlc {
enum class Backend { Auto, Scalar, AVX2, AVX2_FMA, AVX512F };
enum class Preset { Fast, Balanced, Ratio, Adaptive };
struct Options {
  uint32_t time_axis = 0;
  uint32_t tile_elements = 8192;
  uint32_t keyframe_interval = 16;
  Preset preset = Preset::Balanced;
  Backend backend = Backend::Auto;
  int predictor =
      -1; // -1 selects automatically; otherwise 0..3 when history permits.
  bool checksum = true;
  uint64_t memory_limit = uint64_t(1) << 30;
};
struct Tensor {
  std::vector<uint64_t> shape;
  std::vector<uint32_t> bits;
};
struct Metadata {
  std::vector<uint64_t> shape;
  uint32_t time_axis;
  uint32_t tile_elements;
  uint32_t keyframe_interval;
  uint64_t elements;
};
std::vector<uint8_t> compress(std::span<const float>, std::span<const uint64_t>,
                              const Options & = {});
Tensor decompress(std::span<const uint8_t>, Backend = Backend::Auto,
                  uint64_t memory_limit = uint64_t(1) << 30);
Metadata inspect(std::span<const uint8_t>, uint64_t memory_limit = uint64_t(1)
                                                                   << 30);
bool available(Backend);
const char *backend_name(Backend);
std::vector<uint8_t> bitshuffle_words(std::span<const uint32_t>,
                                      Backend = Backend::Auto);
std::vector<uint32_t> bitunshuffle_words(std::span<const uint8_t>,
                                         size_t elements,
                                         Backend = Backend::Auto);
void predict_xor_words(std::span<const uint32_t> input,
                       std::span<const uint32_t> previous,
                       std::span<const uint32_t> previous2,
                       std::span<uint32_t> output, int predictor,
                       Backend backend = Backend::Auto);
} // namespace vtlc
#include "c_api.h"
