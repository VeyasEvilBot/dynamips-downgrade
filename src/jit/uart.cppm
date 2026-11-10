module;

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <string_view>

#include <dynamips/macros.h>

export module jit.uart;

import jit.memory;

export namespace jit {

// Big-endian word registers: TX at +0, TX-ready status at +4.
class uart {
public:
  static constexpr std::size_t capacity = 1024;

  explicit uart(memory::physical_address base) noexcept
      : base_(base), endpoint_{this, read, write} {}

  uart(const uart &) = delete;
  fn operator=(const uart &) -> uart & = delete;

  fn endpoint() noexcept -> memory::device & { return endpoint_; }
  fn output() const noexcept -> std::string_view {
    return {bytes_.data(), size_};
  }

private:
  memory::physical_address base_;
  std::array<char, capacity> bytes_;
  std::size_t size_ = 0;
  memory::device endpoint_;

  static fn read(void *context, memory::device_access access) noexcept
      -> std::expected<std::uint64_t, memory::device_error> {
    const let &self = *static_cast<uart *>(context);
    if (access.size != memory::access_size::word)
      return std::unexpected(memory::device_error::unsupported_size);
    if (access.address.value < self.base_.value ||
        access.address.value - self.base_.value != 4)
      return std::unexpected(memory::device_error::bus);
    std::uint32_t value = self.size_ < capacity ? 1 : 0;
    if (access.order == std::endian::little)
      value = std::byteswap(value);
    return value;
  }

  static fn write(void *context, memory::device_access access,
                   std::uint64_t value) noexcept
      -> std::expected<void, memory::device_error> {
    let &self = *static_cast<uart *>(context);
    if (access.size != memory::access_size::word)
      return std::unexpected(memory::device_error::unsupported_size);
    if (access.address.value != self.base_.value || self.size_ == capacity)
      return std::unexpected(memory::device_error::bus);
    let word = static_cast<std::uint32_t>(value);
    if (access.order == std::endian::little)
      word = std::byteswap(word);
    self.bytes_[self.size_] = static_cast<char>(word & 0xff);
    self.size_++;
    return {};
  }
};
} // namespace jit
