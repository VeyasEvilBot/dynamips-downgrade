extern "C" {
#include "utils.h"
}

#include <cerrno>
#include <cstddef>

#include <dynamips/macros.h>

import host_memory;

cfn m_read_file(const char *filename, unsigned char **buffer,
                std::size_t *length) -> int {
  let loaded = dynamips::host::read_file(filename, buffer != nullptr);
  if (!loaded) {
    if (buffer)
      *buffer = nullptr;
    errno = loaded.error().value();
    return -1;
  }
  if (length)
    *length = loaded->size;
  if (buffer)
    *buffer = loaded->bytes.release();
  return 0;
}

cfn m_memalign(std::size_t boundary, std::size_t size) -> void * {
  let allocated = dynamips::host::allocate_aligned(boundary, size);
  if (!allocated)
    return nullptr;
  return allocated->release();
}
