module;

#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

#include <dynamips/macros.h>

export module devices.byte_memory;

import jit.devices;
import jit.memory;

export namespace dynamips::devices {
// Borrowed backing outlives the endpoint. Reset preserves its contents.
class byte_memory {
public:
  byte_memory(jit::memory::physical_address base,
              std::span<std::byte> bytes) noexcept
      : base_(base), bytes_(bytes), endpoint_{this, read, write} {}

  byte_memory(const byte_memory &) = delete;
  fn operator=(const byte_memory &) -> byte_memory & = delete;

  fn descriptor() noexcept -> jit::devices::descriptor {
    using jit::devices::widths;
    return {&endpoint_,
            widths::byte | widths::half | widths::word | widths::doubleword,
            std::endian::big};
  }

private:
  jit::memory::physical_address base_;
  std::span<std::byte> bytes_;
  jit::memory::device endpoint_;

  fn region(jit::memory::device_access access) noexcept
      -> std::expected<std::span<std::byte>, jit::memory::device_error> {
    const let size = static_cast<unsigned>(access.size);
    if (size != 1 && size != 2 && size != 4 && size != 8)
      return std::unexpected(jit::memory::device_error::unsupported_size);
    if (access.address.value < base_.value)
      return std::unexpected(jit::memory::device_error::bus);
    const let offset = access.address.value - base_.value;
    if (offset > bytes_.size() || size > bytes_.size() - offset)
      return std::unexpected(jit::memory::device_error::bus);
    return bytes_.subspan(static_cast<std::size_t>(offset), size);
  }

  static fn read(void *context, jit::memory::device_access access) noexcept
      -> std::expected<std::uint64_t, jit::memory::device_error> {
    const let bytes = static_cast<byte_memory *>(context)->region(access);
    if (!bytes)
      return std::unexpected(bytes.error());
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < bytes->size(); i++) {
      const let shift =
          8 * (access.order == std::endian::little ? i : bytes->size() - 1 - i);
      value |= std::to_integer<std::uint64_t>((*bytes)[i]) << shift;
    }
    return value;
  }

  static fn write(void *context, jit::memory::device_access access,
                   std::uint64_t value) noexcept
      -> std::expected<void, jit::memory::device_error> {
    const let bytes = static_cast<byte_memory *>(context)->region(access);
    if (!bytes)
      return std::unexpected(bytes.error());
    for (std::size_t i = 0; i < bytes->size(); i++) {
      const let shift =
          8 * (access.order == std::endian::little ? i : bytes->size() - 1 - i);
      (*bytes)[i] = static_cast<std::byte>((value >> shift) & 0xff);
    }
    return {};
  }
};
} // namespace dynamips::devices
