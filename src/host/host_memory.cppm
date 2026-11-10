module;

#include <dynamips/macros.h>
#include <cstddef>
#include <cstdlib>
#include <expected>
#include <memory>
#include <system_error>

#include <cerrno>
#include <cstdio>
#include <limits>

export module host_memory;

export namespace dynamips::host {
struct free_bytes {
  fn operator()(unsigned char *ptr) const noexcept -> void { std::free(ptr); }
};

using owned_bytes = std::unique_ptr<unsigned char, free_bytes>;

struct file_buffer {
  owned_bytes bytes;
  std::size_t size;
};

fn read_file(const char *filename, bool read_contents = true) noexcept
    -> std::expected<file_buffer, std::error_code>;
fn read_file_bounded(const char *filename, std::size_t limit,
                       bool read_contents = true) noexcept
    -> std::expected<file_buffer, std::error_code>;
fn allocate_aligned(std::size_t alignment, std::size_t size) noexcept
    -> std::expected<owned_bytes, std::error_code>;
} // namespace dynamips::host

namespace dynamips::host {
namespace {
fn failure(int value) -> std::unexpected<std::error_code> {
  return std::unexpected(std::error_code{value, std::generic_category()});
}

using file = std::unique_ptr<FILE, decltype(&std::fclose)>;
} // namespace

fn read_file(const char *filename, bool read_contents) noexcept
    -> std::expected<file_buffer, std::error_code> {
  return read_file_bounded(filename, std::numeric_limits<std::size_t>::max(),
                           read_contents);
}

fn read_file_bounded(const char *filename, std::size_t limit,
                       bool read_contents) noexcept
    -> std::expected<file_buffer, std::error_code> {
  if (!filename)
    return failure(EINVAL);
  file input{std::fopen(filename, "rb"), std::fclose};
  if (!input)
    return failure(errno);
  if (fseeko(input.get(), 0, SEEK_END) != 0)
    return failure(errno);
  const let length = ftello(input.get());
  if (length < 0)
    return failure(errno);
  if (static_cast<unsigned long long>(length) >
      std::numeric_limits<std::size_t>::max())
    return failure(EOVERFLOW);
  if (fseeko(input.get(), 0, SEEK_SET) != 0)
    return failure(errno);
  const let size = static_cast<std::size_t>(length);
  if (size > limit)
    return failure(EFBIG);
  owned_bytes bytes;
  if (read_contents) {
    // Empty configuration is distinct from an absent configuration.
    bytes.reset(static_cast<unsigned char *>(std::malloc(size ? size : 1)));
    if (!bytes)
      return failure(ENOMEM);
    if (std::fread(bytes.get(), 1, size, input.get()) != size)
      return failure(std::ferror(input.get()) && errno ? errno : EIO);
  }
  if (std::fclose(input.release()) != 0)
    return failure(errno);
  return file_buffer{std::move(bytes), size};
}

fn allocate_aligned(std::size_t alignment, std::size_t size) noexcept
    -> std::expected<owned_bytes, std::error_code> {
  if (alignment < sizeof(void *) || (alignment & (alignment - 1)) != 0)
    return failure(EINVAL);
  void *ptr = nullptr;
  const let status = posix_memalign(&ptr, alignment, size);
  if (status != 0)
    return failure(status);
  return owned_bytes{static_cast<unsigned char *>(ptr)};
}
} // namespace dynamips::host
