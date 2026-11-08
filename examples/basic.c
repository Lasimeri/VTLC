#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <vtlc/c_api.h>
int main(void) {
  float input[] = {1, 2, 3, 1, 2, 3};
  uint64_t shape[] = {2, 3}, size = 0, count = 0, *decoded_shape = 0;
  unsigned rank = 0;
  unsigned char *bytes = 0, *decoded = 0;
  assert(vtlc_compress_c(input, shape, 2, 0, &bytes, &size) == 0);
  assert(vtlc_decompress_c(bytes, size, &decoded, &count, &decoded_shape,
                           &rank) == 0);
  assert(count == 6 && !memcmp(input, decoded, sizeof(input)));
  vtlc_free(bytes);
  vtlc_free(decoded);
  vtlc_free(decoded_shape);
  return 0;
}
