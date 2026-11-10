module;

#include <dynamips/macros.h>
#include <cerrno>
#include <cstdio>
#include <print>
#include <expected>
#include <fcntl.h>
#include <system_error>
#include <unistd.h>
#include <utility>

export module native_fd;

export namespace dynamips::host {
class file_descriptor {
  int value_;

public:
  explicit file_descriptor(int value = -1) noexcept : value_(value) {}

  file_descriptor(const file_descriptor &) = delete;
  fn operator=(const file_descriptor &) -> file_descriptor & = delete;

  file_descriptor(file_descriptor &&other) noexcept : value_(other.release()) {}

  fn operator=(file_descriptor &&other) noexcept -> file_descriptor & {
    file_descriptor previous{std::exchange(value_, other.release())};
    return *this;
  }

  ~file_descriptor() {
    if (value_ >= 0 && close(value_) != 0) {
      try {
        std::println(stderr, "close FD {}: error {}", value_, errno);
      } catch (...) {
      } // Diagnostics must not escape destruction.
    }
  }

  fn get() const noexcept -> int { return value_; }

  fn release() noexcept -> int { return std::exchange(value_, -1); }
};

inline fn duplicate_fd(int borrowed) noexcept
    -> std::expected<file_descriptor, std::error_code> {
  const let value = fcntl(borrowed, F_DUPFD_CLOEXEC, 0);
  if (value < 0)
    return std::unexpected(std::error_code{errno, std::generic_category()});
  return file_descriptor{value};
}
} // namespace dynamips::host
