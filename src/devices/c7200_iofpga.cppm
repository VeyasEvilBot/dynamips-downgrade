module;
#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <utility>
#include <dynamips/macros.h>
export module devices.c7200_iofpga;
import jit.devices;
import jit.memory;
import jit.uart;

export namespace dynamips::devices {
class c7200_iofpga {
public:
  c7200_iofpga() noexcept : endpoint_{this, read, write} {}

  c7200_iofpga(const c7200_iofpga &) = delete;
  fn operator=(const c7200_iofpga &) -> c7200_iofpga & = delete;

  fn descriptor() noexcept -> jit::devices::descriptor {
    using jit::devices::widths;
    return {&endpoint_,
            widths::byte | widths::half | widths::word,
            std::endian::big,
            {this,
             [](void *context) noexcept {
               static_cast<c7200_iofpga *>(context)->reset();
             },
             nullptr}};
  }
  fn reset() noexcept -> void {
    console_.reset();
    auxiliary_.reset();
    control_ = 0;
    interrupt_mask_ = 0;
  }

  template <typename Sink> fn drain(bool auxiliary, Sink &&sink) noexcept {
    return (auxiliary ? auxiliary_ : console_).drain(std::forward<Sink>(sink));
  }
  fn receive(std::span<const std::byte> bytes) noexcept -> std::size_t {
    return console_.receive(bytes);
  }

private:
  jit::uart console_{{0}}, auxiliary_{{0}};
  std::uint32_t control_ = 0;
  std::uint8_t interrupt_mask_ = 0;
  jit::memory::device endpoint_;

  static fn valid_size(jit::memory::access_size size) noexcept -> bool {
    using jit::memory::access_size;
    return size == access_size::byte || size == access_size::half ||
           size == access_size::word;
  }

  static fn uart_access(std::uint64_t offset) noexcept
      -> jit::memory::device_access {
    return {{offset}, std::endian::big, jit::memory::access_size::word};
  }

  static fn read(void *context, jit::memory::device_access access) noexcept
      -> std::expected<std::uint64_t, jit::memory::device_error> {
    if (!valid_size(access.size))
      return std::unexpected(jit::memory::device_error::unsupported_size);
    let &self = *static_cast<c7200_iofpga *>(context);
    const let offset = access.address.value - 0x1e840000;
    switch (offset) {
    case 0x204:
      return self.control_ | 4u;
    case 0x224:
    case 0x234:
    case 0x244:
    case 0x3a0:
      return 0;
    case 0x23c:
      return 0x2704;
    case 0x40c:
    case 0x44c: {
      let &uart =
          (offset == 0x40c ? self.console_ : self.auxiliary_).endpoint();
      const let status = uart.read(uart.context, uart_access(4));
      if (!status)
        return std::unexpected(status.error());
      return ((*status & 1) ? 4u : 0u) | ((*status & 2) ? 1u : 0u);
    }
    case 0x41c:
    case 0x45c: {
      let &uart =
          (offset == 0x41c ? self.console_ : self.auxiliary_).endpoint();
      return uart.read(uart.context, uart_access(8));
    }
    case 0x42c:
      return (self.console_.pending_input() ? 2u : 0u) |
             (self.auxiliary_.pending_input() ? 0x20u : 0u);
    case 0x404:
    case 0x414:
    case 0x424:
    case 0x434:
    case 0x43c:
    case 0x444:
    case 0x454:
    case 0x46c:
    case 0x474:
    case 0x47c:
      return 0;
    default:
      return std::unexpected(jit::memory::device_error::bus);
    }
  }

  static fn write(void *context, jit::memory::device_access access,
                   std::uint64_t value) noexcept
      -> std::expected<void, jit::memory::device_error> {
    if (!valid_size(access.size))
      return std::unexpected(jit::memory::device_error::unsupported_size);
    let &self = *static_cast<c7200_iofpga *>(context);
    const let offset = access.address.value - 0x1e840000;
    switch (offset) {
    case 0x204:
      self.control_ = static_cast<std::uint32_t>(value);
      return {};
    case 0x42c:
      self.interrupt_mask_ = static_cast<std::uint8_t>(value);
      return {};
    case 0x41c:
    case 0x45c: {
      let &uart =
          (offset == 0x41c ? self.console_ : self.auxiliary_).endpoint();
      return uart.write(uart.context, uart_access(0), value);
    }
    default:
      const let known = read(context, access);
      if (!known)
        return std::unexpected(known.error());
      return {};
    }
  }
};
} // namespace dynamips::devices
