#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <print>
#include <system_error>
#include <unistd.h>
#include <utility>

#include <dynamips/macros.h>

import native_fd;
import host_memory;

extern "C" fn host_compat_check(const char *, const unsigned char *,
                                 std::size_t) -> int;

namespace {
fn check(bool condition, const char *message) -> bool {
  if (!condition)
    std::println(stderr, "host test: {}", message);
  return condition;
}
} // namespace

fn main() -> int {
  namespace host = dynamips::host;
  char filename[] = "/tmp/dynamips-host-XXXXXX";
  host::file_descriptor descriptor{mkstemp(filename)};
  if (!check(descriptor.get() >= 0, "create file fixture"))
    return 1;
  const let result = [&]() -> int {
    constexpr unsigned char payload[]{'n', 'a', 't', 'i', 'v',
                                      'e', 0,   'i', 'o', '\n'};
    if (!check(::write(descriptor.get(), payload, sizeof(payload)) ==
                   sizeof(payload),
               "write real binary fixture"))
      return 1;
    let contents = host::read_file(filename);
    const let metadata = host::read_file(filename, false);
    if (!check(contents && contents->size == sizeof(payload) &&
                   contents->bytes &&
                   std::memcmp(contents->bytes.get(), payload,
                               sizeof(payload)) == 0 &&
                   metadata && metadata->size == sizeof(payload) &&
                   !metadata->bytes,
               "C++ file ownership, binary contents and metadata") ||
        !check(host_compat_check(filename, payload, sizeof(payload)) == 0,
               "real C caller preserves buffer ownership, errno and allocation "
               "ABI"))
      return 1;
    const let bounded = host::read_file_bounded(filename, sizeof(payload));
    const let too_large =
        host::read_file_bounded(filename, sizeof(payload) - 1);
    if (!check(bounded && bounded->size == sizeof(payload) && !too_large &&
                   too_large.error() == std::errc::file_too_large,
               "single-open bounded reads enforce the limit before allocating "
               "contents"))
      return 1;
    let duplicate = host::duplicate_fd(descriptor.get());
    if (!check(duplicate && duplicate->get() != descriptor.get() &&
                   (fcntl(duplicate->get(), F_GETFD) & FD_CLOEXEC) != 0,
               "duplicate owns a distinct close-on-exec descriptor"))
      return 1;
    const let old = duplicate->get();
    {
      host::file_descriptor moved{std::move(*duplicate)};
      if (!check(duplicate->get() == -1 && moved.get() == old,
                 "moving transfers descriptor ownership"))
        return 1;
      descriptor = host::file_descriptor{};
      std::array<unsigned char, sizeof(payload)> readback{};
      if (!check(lseek(moved.get(), 0, SEEK_SET) == 0 &&
                     ::read(moved.get(), readback.data(), readback.size()) ==
                         static_cast<ssize_t>(readback.size()) &&
                     std::memcmp(readback.data(), payload, sizeof(payload)) ==
                         0,
                 "duplicate survives closing the original"))
        return 1;
      if (!check(ftruncate(moved.get(), 0) == 0, "truncate empty fixture"))
        return 1;
    }
    errno = 0;
    if (!check(fcntl(old, F_GETFD) == -1 && errno == EBADF,
               "RAII closes the owned descriptor"))
      return 1;
    const let empty = host::read_file_bounded(filename, 0);
    if (!check(empty && empty->size == 0 && empty->bytes &&
                   host_compat_check(filename, payload, 0) == 0,
               "empty files remain distinct from absent files for both APIs"))
      return 1;
    const let invalid = host::read_file(nullptr);
    const let invalid_fd = host::duplicate_fd(-1);
    const let invalid_alignment = host::allocate_aligned(3, 64);
    if (!check(!invalid && invalid.error() == std::errc::invalid_argument &&
                   !invalid_fd &&
                   invalid_fd.error() == std::errc::bad_file_descriptor &&
                   !invalid_alignment &&
                   invalid_alignment.error() == std::errc::invalid_argument,
               "C++ failures retain typed host errors"))
      return 1;
    let memory = host::allocate_aligned(4096, 8192);
    if (!check(memory &&
                   reinterpret_cast<std::uintptr_t>(memory->get()) % 4096 == 0,
               "aligned C++ allocation retains RAII ownership"))
      return 1;
    std::memset(memory->get(), 0xa5, 8192);
    let pages = host::map_zeroed(8193);
    if (!check(pages && pages->get()[0] == 0 && pages->get()[8192] == 0 &&
                   !host::map_zeroed(0),
               "anonymous RAM begins zeroed and rejects an empty mapping"))
      return 1;
    pages->get()[8192] = 99;
    let moved_pages = std::move(*pages);
    if (!check(!*pages && moved_pages.get()[8192] == 99,
               "moving anonymous RAM transfers its mapping lifetime"))
      return 1;
    return 0;
  }();
  const int removed = unlink(filename);
  if (result != 0 || !check(removed == 0, "remove file fixture"))
    return 1;
  std::println("host: direct C++ helpers and C compatibility checks passed");
}
