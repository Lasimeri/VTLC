#include "vtlc/c_api.h"
#include <assert.h>
#include <string.h>
int main(void) {
  float x[] = {1, 2, 3, 1, 2, 3};
  uint64_t shape[] = {2, 3}, size = 0, n = 0, *restored_shape = 0;
  unsigned char *compressed = 0, *restored = 0;
  unsigned dims = 0;
  assert(!vtlc_compress_c(x, shape, 2, 0, &compressed, &size));
  assert(!vtlc_decompress_c(compressed, size, &restored, &n, &restored_shape,
                            &dims));
  assert(n == 6 && dims == 2 && !memcmp(x, restored, sizeof(x)));
  assert(restored_shape[0] == 2 && restored_shape[1] == 3);
  vtlc_free(compressed);
  vtlc_free(restored);
  vtlc_free(restored_shape);
  assert(vtlc_compress_c(x, 0, 2, 0, &compressed, &size));
  return 0;
}
