#ifndef VTLC_C_API_H
#define VTLC_C_API_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int vtlc_compress_c(const float *input, const uint64_t *shape,
                    unsigned dimensions, unsigned time_axis,
                    unsigned char **output, uint64_t *output_size);
int vtlc_decompress_c(const unsigned char *input, uint64_t input_size,
                      unsigned char **output, uint64_t *elements,
                      uint64_t **shape, unsigned *dimensions);
void vtlc_free(void *ptr);
const char *vtlc_error(void);
#ifdef __cplusplus
}
#endif
#endif
