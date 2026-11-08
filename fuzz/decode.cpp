#include "vtlc/codec.hpp"
#include "vtlc/streaming.hpp"
#include <cstddef>
#include <cstdint>
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  try {
    (void)vtlc::decompress({data, size}, vtlc::Backend::Scalar, 1 << 20);
  } catch (...) {
  }
  try {
    vtlc::StreamDecoder stream({data, size}, vtlc::Backend::Scalar, 1 << 20);
    std::vector<uint32_t> frame;
    for (int i = 0; i < 16 && stream.decode_next(frame); i++) {
    }
  } catch (...) {
  }
  return 0;
}
