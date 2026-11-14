module;
#include <array>
#include <bit>
#include <cstdint>
#include <expected>
#include <dynamips/macros.h>
export module devices.clpd6729;
import jit.devices;
import jit.memory;
import devices.gt64120;

export namespace dynamips::devices {
// Two empty sockets. Configuration is retained, but card memory and IRQ wiring
// require actual card/device implementations.
class clpd6729 {
public:
  explicit clpd6729(
      jit::memory::physical_address base = {0x100000000ULL}) noexcept
      : base_(base), endpoint_{this, read, write} {
    reset();
  }

  clpd6729(const clpd6729 &) = delete;
  fn operator=(const clpd6729 &) -> clpd6729 & = delete;

  fn reset() noexcept -> void {
    registers_ = {};
    extended_ = {};
    index_ = 0;
    command_ = 0x80;
    io_base_ = 0;
    bar_probe_ = false;
    interrupt_line_ = 0;
    identification_high_ = true;
    registers_[0x1e] = 2;
    for (unsigned socket = 0; socket < 2; socket++) {
      const let first = socket * 64;
      registers_[first + 0x3a] = registers_[first + 0x3d] = 1;
      registers_[first + 0x3b] = 5;
      registers_[first + 0x3e] = 0x14;
    }
  }
  fn descriptor() noexcept -> jit::devices::descriptor {
    return {&endpoint_,
            jit::devices::widths::byte,
            std::endian::little,
            {this,
             [](void *context) noexcept {
               static_cast<clpd6729 *>(context)->reset();
             },
             nullptr}};
  }
  fn configuration() noexcept -> pci_function {
    return {0, 2, 16, 0, this, config_read, config_write};
  }

private:
  jit::memory::physical_address base_;
  std::array<std::uint8_t, 128> registers_{};
  std::array<std::array<std::uint8_t, 64>, 2> extended_{};
  std::uint8_t index_ = 0, interrupt_line_ = 0;
  std::uint16_t command_ = 0x80;
  std::uint32_t io_base_ = 0;
  bool bar_probe_ = false, identification_high_ = true;
  jit::memory::device endpoint_;

  static fn config_read(void *context, std::uint8_t reg) noexcept
      -> std::uint32_t {
    const let &self = *static_cast<clpd6729 *>(context);
    switch (reg) {
    case 0:
      return 0x11001013;
    case 4:
      return 0x04000000u | self.command_;
    case 8:
      return 0x06050000;
    case 0x10:
      return self.bar_probe_ ? 0xfffffffdu : self.io_base_ | 1u;
    case 0x3c:
      return 0x100u | self.interrupt_line_;
    default:
      return 0;
    }
  }

  static fn config_write(void *context, std::uint8_t reg,
                          std::uint32_t value) noexcept -> void {
    let &self = *static_cast<clpd6729 *>(context);
    switch (reg) {
    case 4:
      self.command_ = static_cast<std::uint16_t>((value & 0x143) | 0x80);
      break;
    case 0x10:
      self.bar_probe_ = value == 0xffffffffu;
      if (!self.bar_probe_)
        self.io_base_ = value & 0xfffffffcu;
      break;
    case 0x3c:
      self.interrupt_line_ = static_cast<std::uint8_t>(value);
      break;
    default:
      break;
    }
  }

  static fn writable_mask(unsigned reg) noexcept -> unsigned {
    if (reg == 2 || reg == 3 || reg == 5 || reg == 7 ||
        (reg >= 8 && reg <= 0x16) || (reg >= 0x18 && reg <= 0x1d) ||
        (reg >= 0x20 && reg <= 0x26) || (reg >= 0x28 && reg <= 0x2e) ||
        (reg >= 0x30 && reg <= 0x35) || reg == 0x37 || reg == 0x39 ||
        reg >= 0x3a)
      return 0xff;
    if (reg == 6)
      return 0xdf;
    if (reg == 0x17)
      return 0x7f;
    if (reg == 0x1e)
      return 0x8f;
    if (reg == 0x36 || reg == 0x38)
      return 0xfe;
    return 0;
  }
  fn extended_register() noexcept -> std::uint8_t * {
    const unsigned socket = index_ >> 6;
    const unsigned selected = registers_[socket * 64 + 0x2e];
    if (selected == 0 || selected == 3 || (selected >= 5 && selected <= 9) ||
        selected == 0x25)
      return &extended_[socket][selected];
    return nullptr;
  }
  fn register_read() noexcept
      -> std::expected<std::uint64_t, jit::memory::device_error> {
    if (index_ >= registers_.size())
      return std::unexpected(jit::memory::device_error::bus);
    const unsigned reg = index_ & 63;
    const unsigned first = index_ & 64;
    switch (reg) {
    case 0:
      return 0x82;
    case 1: {
      const let power = registers_[first + 2];
      return 0x80u | ((power & 0x30) == 0x10 ? 0x40u : 0u);
    }
    case 4: {
      const let result = registers_[index_];
      registers_[index_] = 0;
      return result;
    }
    case 0x17:
      return registers_[index_] | 0x80u;
    case 0x1e:
      return registers_[0x1e];
    case 0x1f: {
      const unsigned value = identification_high_ ? 0xc0u : 0;
      identification_high_ = !identification_high_;
      return value;
    }
    case 0x2f:
      if (const let *value = extended_register())
        return *value;
      return std::unexpected(jit::memory::device_error::bus);
    default:
      if (writable_mask(reg))
        return registers_[index_];
      return std::unexpected(jit::memory::device_error::bus);
    }
  }
  fn register_write(std::uint8_t value) noexcept
      -> std::expected<void, jit::memory::device_error> {
    if (index_ >= registers_.size())
      return std::unexpected(jit::memory::device_error::bus);
    const unsigned reg = index_ & 63;
    if (reg == 0 || reg == 1 || reg == 4)
      return {};
    if (reg == 0x1f) {
      identification_high_ = true;
      return {};
    }
    if (reg == 0x2f) {
      if (let *target = extended_register()) {
        const unsigned selected = registers_[(index_ & 64) + 0x2e];
        *target =
            static_cast<std::uint8_t>(value & (selected == 3      ? 0x3f
                                               : selected == 0x25 ? 0xc0
                                                                  : 0xff));
        return {};
      }
      return std::unexpected(jit::memory::device_error::bus);
    }
    if (reg == 2 && (extended_[index_ >> 6][3] & 1))
      value = static_cast<std::uint8_t>((value & ~0x10u) |
                                        (registers_[index_] & 0x10));
    if (reg == 0x1e && (value & 4))
      return std::unexpected(jit::memory::device_error::bus);
    const let mask = writable_mask(reg);
    if (!mask)
      return std::unexpected(jit::memory::device_error::bus);
    registers_[reg == 0x1e ? 0x1e : index_] =
        static_cast<std::uint8_t>(value & mask);
    return {};
  }
  fn port(jit::memory::device_access access) const noexcept
      -> std::expected<unsigned, jit::memory::device_error> {
    if (access.size != jit::memory::access_size::byte)
      return std::unexpected(jit::memory::device_error::unsupported_size);
    if (bar_probe_ || !(command_ & 1) || access.address.value < base_.value ||
        access.address.value - base_.value >= 0x1000)
      return std::unexpected(jit::memory::device_error::bus);
    const let logical = (access.address.value - base_.value) ^ 3;
    if (logical == io_base_)
      return 0;
    if (logical == std::uint64_t{io_base_} + 1)
      return 1;
    return std::unexpected(jit::memory::device_error::bus);
  }

  static fn read(void *context, jit::memory::device_access access) noexcept
      -> std::expected<std::uint64_t, jit::memory::device_error> {
    let &self = *static_cast<clpd6729 *>(context);
    const let selected = self.port(access);
    if (!selected)
      return std::unexpected(selected.error());
    return *selected == 0
               ? std::expected<std::uint64_t,
                               jit::memory::device_error>{self.index_}
               : self.register_read();
  }

  static fn write(void *context, jit::memory::device_access access,
                   std::uint64_t value) noexcept
      -> std::expected<void, jit::memory::device_error> {
    let &self = *static_cast<clpd6729 *>(context);
    const let selected = self.port(access);
    if (!selected)
      return std::unexpected(selected.error());
    if (*selected == 0) {
      self.index_ = static_cast<std::uint8_t>(value);
      return {};
    }
    return self.register_write(static_cast<std::uint8_t>(value));
  }
};
} // namespace dynamips::devices
