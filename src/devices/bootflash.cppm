module;
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <dynamips/macros.h>
export module devices.bootflash;
import jit.devices;
import jit.memory;

export namespace dynamips::devices {
// Four interleaved 28F016SA byte-mode chips. Backing outlives the endpoint;
// program and erase complete synchronously. Reset preserves stored contents.
class bootflash {
public:
  static constexpr std::size_t size = 8 * 1024 * 1024;
  static constexpr std::size_t block_size = 4 * 0x10000;

  bootflash(jit::memory::physical_address base,
            std::span<std::byte> bytes) noexcept
      : base_(base), bytes_(bytes), endpoint_{this, read, write} {}

  bootflash(const bootflash &) = delete;
  fn operator=(const bootflash &) -> bootflash & = delete;

  fn reset() noexcept -> void { chips_ = {}; }
  fn descriptor() noexcept -> jit::devices::descriptor {
    using jit::devices::widths;
    return {&endpoint_,
            widths::byte | widths::half | widths::word | widths::doubleword,
            std::endian::big,
            {this,
             [](void *context) noexcept {
               static_cast<bootflash *>(context)->reset();
             },
             nullptr}};
  }

private:
  enum class mode : std::uint8_t { array, identifier, status, program, erase };

  struct chip {
    mode state = mode::array;
    std::uint8_t status = 0x80;
  };

  jit::memory::physical_address base_;
  std::span<std::byte> bytes_;
  std::array<chip, 4> chips_{};
  jit::memory::device endpoint_;

  fn region(jit::memory::device_access access) const noexcept
      -> std::expected<std::size_t, jit::memory::device_error> {
    const unsigned width = static_cast<unsigned>(access.size);
    if (width != 1 && width != 2 && width != 4 && width != 8)
      return std::unexpected(jit::memory::device_error::unsupported_size);
    if (bytes_.size() != size || base_.value > UINT64_MAX - (size - 1) ||
        access.address.value < base_.value)
      return std::unexpected(jit::memory::device_error::bus);
    const let offset = access.address.value - base_.value;
    if (offset > size || width > size - offset)
      return std::unexpected(jit::memory::device_error::bus);
    return static_cast<std::size_t>(offset);
  }
  fn read_byte(std::size_t offset) const noexcept
      -> std::expected<std::uint8_t, jit::memory::device_error> {
    const let &current = chips_[offset & 3];
    switch (current.state) {
    case mode::array:
      return std::to_integer<std::uint8_t>(bytes_[offset]);
    case mode::identifier:
      if (offset / 4 == 0)
        return 0x89;
      if (offset / 4 == 1)
        return 0xa0;
      return std::unexpected(jit::memory::device_error::bus);
    default:
      return current.status;
    }
  }
  fn write_byte(std::size_t offset, std::uint8_t value) noexcept
      -> std::expected<void, jit::memory::device_error> {
    let &current = chips_[offset & 3];
    if (current.state == mode::program) {
      const let old = std::to_integer<std::uint8_t>(bytes_[offset]);
      if ((old & value) != value)
        current.status |= 0x10;
      bytes_[offset] = static_cast<std::byte>(old & value);
      current.state = mode::status;
      return {};
    }
    if (current.state == mode::erase) {
      current.state = mode::status;
      if (value != 0xd0) {
        current.status |= 0x30;
        return std::unexpected(jit::memory::device_error::bus);
      }
      const let block = offset & ~(block_size - 1);
      for (std::size_t position = block + (offset & 3);
           position < block + block_size; position += 4)
        bytes_[position] = std::byte{0xff};
      return {};
    }
    switch (value) {
    case 0xff:
      current.state = mode::array;
      return {};
    case 0x90:
      current.state = mode::identifier;
      return {};
    case 0x70:
      current.state = mode::status;
      return {};
    case 0x50:
      current.status &= 0xc7;
      return {};
    case 0x40:
    case 0x10:
      current.state = mode::program;
      return {};
    case 0x20:
      current.state = mode::erase;
      return {};
    default:
      return std::unexpected(jit::memory::device_error::bus);
    }
  }

  static fn read(void *context, jit::memory::device_access access) noexcept
      -> std::expected<std::uint64_t, jit::memory::device_error> {
    const let &self = *static_cast<bootflash *>(context);
    const let offset = self.region(access);
    if (!offset)
      return std::unexpected(offset.error());
    const unsigned width = static_cast<unsigned>(access.size);
    std::uint64_t result = 0;
    for (unsigned i = 0; i < width; i++) {
      const let byte = self.read_byte(*offset + i);
      if (!byte)
        return std::unexpected(byte.error());
      const let shift =
          8 * (access.order == std::endian::little ? i : width - i - 1);
      result |= std::uint64_t{*byte} << shift;
    }
    return result;
  }

  static fn write(void *context, jit::memory::device_access access,
                   std::uint64_t value) noexcept
      -> std::expected<void, jit::memory::device_error> {
    let &self = *static_cast<bootflash *>(context);
    const let offset = self.region(access);
    if (!offset)
      return std::unexpected(offset.error());
    const unsigned width = static_cast<unsigned>(access.size);
    for (unsigned i = 0; i < width; i++) {
      const let shift =
          8 * (access.order == std::endian::little ? i : width - i - 1);
      const let result = self.write_byte(
          *offset + i, static_cast<std::uint8_t>(value >> shift));
      if (!result)
        return std::unexpected(result.error());
    }
    return {};
  }
};
} // namespace dynamips::devices
