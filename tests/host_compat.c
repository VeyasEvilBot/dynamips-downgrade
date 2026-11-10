#include "utils.h"
#include <stdint.h>

int host_compat_check(const char *filename, const unsigned char *expected,
                      size_t expected_length) {
  unsigned char *buffer = NULL;
  size_t length = 0;
  if (m_read_file(filename, &buffer, &length) != 0)
    return 1;
  const int mismatch = !buffer || length != expected_length ||
                       memcmp(buffer, expected, expected_length) != 0;
  free(buffer);
  if (mismatch || m_read_file(filename, NULL, &length) != 0 ||
      length != expected_length)
    return 1;
  buffer = (unsigned char *)1;
  length = 123;
  errno = 0;
  if (m_read_file(NULL, &buffer, &length) != -1 || buffer != NULL ||
      errno != EINVAL || length != 123)
    return 1;
  void *aligned = m_memalign(4096, 8192);
  if (!aligned || ((uintptr_t)aligned % 4096) != 0) {
    free(aligned);
    return 1;
  }
  free(aligned);
  if (m_memalign(3, 16) != NULL)
    return 1;
  return 0;
}
