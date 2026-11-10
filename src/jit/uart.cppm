module;

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <span>
#include <string_view>
#include <utility>

#include <dynamips/macros.h>

export module jit.uart;

import jit.memory;
import jit.fifo;

export namespace jit {

// Big-endian words: TX +0, status +4 (TX-ready bit 0, RX-ready bit 1), RX +8.
class uart {
public:
  static constexpr std::size_t capacity = 1024;

  explicit uart(memory::physical_address base) noexcept
      : base_(base), endpoint_{this, read, write} {}

  uart(const uart &) = delete;
  fn operator=(const uart &) -> uart & = delete;

  fn endpoint() noexcept -> memory::device & { return endpoint_; }
  fn reset() noexcept -> void {
    tx_ = {};
    rx_ = {};
  }

  fn pending_input() const noexcept -> std::size_t { return rx_.size(); }

  fn pending() const noexcept -> std::size_t { return tx_.size(); }
  fn receive(std::span<const std::byte> bytes) noexcept -> std::size_t {
    return rx_.write(bytes);
  }

  template <typename Sink> fn drain(Sink &&sink) noexcept {
    return tx_.drain(std::forward<Sink>(sink));
  }

  // Diagnostic snapshot, valid until the next output() call.
  fn output() const noexcept -> std::string_view {
    const let parts = tx_.segments();
    std::memcpy(snapshot_.data(), parts[0].data(), parts[0].size());
    std::memcpy(snapshot_.data() + parts[0].size(), parts[1].data(),
                parts[1].size());
    return {snapshot_.data(), tx_.size()};
  }

private:
  memory::physical_address base_;
  byte_fifo<capacity> tx_;
  byte_fifo<capacity> rx_;
  mutable std::array<char, capacity> snapshot_;
  memory::device endpoint_;

  static fn read(void *context, memory::device_access access) noexcept
      -> std::expected<std::uint64_t, memory::device_error> {
    let &self = *static_cast<uart *>(context);
    if (access.size != memory::access_size::word)
      return std::unexpected(memory::device_error::unsupported_size);
    if (access.address.value < self.base_.value)
      return std::unexpected(memory::device_error::bus);
    const let offset = access.address.value - self.base_.value;
    std::uint32_t value;
    if (offset == 4)
      value = (self.tx_.available() ? 1u : 0u) | (self.rx_.size() ? 2u : 0u);
    else if (offset == 8) {
      const let byte = self.rx_.pop();
      if (!byte)
        return std::unexpected(memory::device_error::bus);
      value = std::to_integer<std::uint32_t>(*byte);
    } else
      return std::unexpected(memory::device_error::bus);
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
    if (access.address.value != self.base_.value || self.tx_.available() == 0)
      return std::unexpected(memory::device_error::bus);
    let word = static_cast<std::uint32_t>(value);
    if (access.order == std::endian::little)
      word = std::byteswap(word);
    static_cast<void>(self.tx_.push(static_cast<std::byte>(word & 0xff)));
    return {};
  }
};
} // namespace jit
