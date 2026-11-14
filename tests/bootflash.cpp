#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <print>
#include <span>
#include <string_view>
#include <vector>
#include <dynamips/macros.h>
import devices.bootflash;
import jit.memory;
import jit.devices;

namespace {
namespace mem = jit::memory;
using flash = dynamips::devices::bootflash;
fn require(bool good, std::string_view message) -> void {
  if (!good) {
    std::println(stderr, "bootflash test: {}", message);
    std::exit(1);
  }
}

struct fixture {
  std::vector<std::byte> bytes =
      std::vector<std::byte>(flash::size, std::byte{0xff});
  flash chip{{0x1a000000}, bytes};
  mem::device &endpoint = *chip.descriptor().endpoint;
  fn access(std::size_t offset, mem::access_size width,
              std::endian order) const -> mem::device_access {
    return {{0x1a000000 + offset}, order, width};
  }
  fn read(std::size_t offset = 0,
            mem::access_size width = mem::access_size::word,
            std::endian order = std::endian::big) -> std::uint64_t {
    const let value =
        endpoint.read(endpoint.context, access(offset, width, order));
    require(value.has_value(), "read flash");
    return *value;
  }
  fn write(std::size_t offset, std::uint64_t value,
             mem::access_size width = mem::access_size::word,
             std::endian order = std::endian::big) -> void {
    require(
        endpoint.write(endpoint.context, access(offset, width, order), value)
            .has_value(),
        "write flash");
  }
};

fn identity_and_lanes() -> void {
  fixture f;
  require(f.read() == 0xffffffff, "erased array reads all ones");
  f.write(0, 0x90909090);
  require(f.read() == 0x89898989 && f.read(4) == 0xa0a0a0a0,
          "four Intel28F016SA byte lanes expose their own manufacturer and "
          "device IDs");
  require(f.read(0, mem::access_size::doubleword) == 0x89898989a0a0a0a0ULL,
          "doubleword reads span two physical flash bus cycles");
  f.write(1, 0xff, mem::access_size::byte);
  require(f.read() == 0x89ff8989 && f.read(4) == 0xa0ffa0a0,
          "byte commands affect only their selected lane");
  f.write(2, 0xffff, mem::access_size::half);
  require(f.read() == 0x89ffffff, "halfword commands affect two lanes only");
  f.chip.reset();
  for (unsigned i = 0; i < 8; i++)
    f.bytes[i] = static_cast<std::byte>(i + 1);
  require(f.read() == 0x01020304 &&
              f.read(0, mem::access_size::word, std::endian::little) ==
                  0x04030201 &&
              f.read(0, mem::access_size::doubleword) == 0x0102030405060708ULL,
          "array byte order follows each access without changing stored data");
}
fn program_and_status() -> void {
  fixture f;
  f.write(0, 0x40404040);
  f.write(0x100, 0x12345678);
  require(
      f.read() == 0x80808080 && f.read() == 0x80808080,
      "program completion leaves stable ready status until another command");
  f.write(0, 0xffffffff);
  require(
      f.read(0x100) == 0x12345678 && f.read(0x104) == 0xffffffff,
      "program writes the supplied address and leaves adjacent data erased");
  f.write(0, 0x10101010);
  f.write(0x100, 0xffffffff);
  require(f.read() == 0x90909090,
          "zero-to-one attempts latch a per-chip program error");
  f.write(0, 0xffffffff);
  require(f.read(0x100) == 0x12345678,
          "program errors cannot change zero bits to one");
  f.write(0, 0x70707070);
  require(f.read(0x100) == 0x90909090,
          "error state survives read-array and read-status changes");
  f.write(0, 0x50505050);
  require(f.read() == 0x80808080,
          "clear-status removes sticky operation errors");
  f.write(2, 0x40, mem::access_size::byte);
  f.write(0x202, 0x88, mem::access_size::byte);
  f.chip.reset();
  require(f.read(0x200) == 0xffff88ff && f.read(0x100) == 0x12345678,
          "program data is not decoded as a command, reset retains contents");
  f.write(0, 0x40404040);
  f.write(0x300, 0x11223344, mem::access_size::word, std::endian::little);
  f.chip.reset();
  require(f.read(0x300) == 0x44332211,
          "little-endian program payload selects the matching physical lanes");
}
fn erase_and_failures() -> void {
  fixture f;
  std::ranges::fill(f.bytes, std::byte{0});
  f.write(0, 0x20, mem::access_size::byte);
  f.write(flash::block_size + 8, 0xd0, mem::access_size::byte);
  f.write(0, 0xff, mem::access_size::byte);
  require(f.read(flash::block_size) == 0xff000000 &&
              f.read(flash::block_size * 2 - 4) == 0xff000000 &&
              f.read(0) == 0 && f.read(flash::block_size * 2) == 0,
          "erase confirmation selects one full64KiB chip block without "
          "touching other lanes or blocks");
  f.write(0, 0x20202020);
  f.write(flash::size - 4, 0xd0d0d0d0);
  require(f.read() == 0x80808080, "completed erase reports ready per chip");
  f.chip.reset();
  require(f.read(flash::size - flash::block_size) == 0xffffffff &&
              f.read(flash::size - 4) == 0xffffffff &&
              f.read(flash::size - flash::block_size - 4) == 0,
          "last flash block erases exactly to the device boundary");
  f.write(0, 0x20, mem::access_size::byte);
  require(!f.endpoint.write(
              f.endpoint.context,
              f.access(0, mem::access_size::byte, std::endian::big), 0x11),
          "invalid erase confirmation is an explicit failure");
  require(f.read(0, mem::access_size::byte) == 0xb0,
          "invalid erase confirmation latches sequence and erase errors");
  f.chip.reset();
  require(f.read() == 0, "invalid erase does not modify array data");
  require(!f.endpoint.write(
              f.endpoint.context,
              f.access(0, mem::access_size::byte, std::endian::big), 0x98),
          "CFI query is not fabricated for an unsupported command set");
  require(!f.endpoint.read(f.endpoint.context,
                           f.access(flash::size, mem::access_size::byte,
                                    std::endian::big)) &&
              !f.endpoint.write(f.endpoint.context,
                                f.access(flash::size - 4,
                                         mem::access_size::doubleword,
                                         std::endian::big),
                                0xffffffffffffffffULL),
          "range is validated before accesses can touch outside backing");
  flash invalid{{0x1a000000}, std::span<std::byte>{}};
  let &endpoint = *invalid.descriptor().endpoint;
  require(
      !endpoint.read(endpoint.context,
                     {{0x1a000000}, std::endian::big, mem::access_size::byte}),
      "invalid backing returns a typed failure");
}
} // namespace

fn main() -> int {
  identity_and_lanes();
  program_and_status();
  erase_and_failures();
  std::println("bootflash: Intel IDs, lane state, program/erase, sticky "
               "status, reset and bounds passed");
}
