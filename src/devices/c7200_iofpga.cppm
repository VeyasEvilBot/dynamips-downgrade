module;
#include <array>
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
import devices.serial_eeprom;
import devices.ds1620;
import devices.c7200_environment;

export namespace dynamips::devices {
class c7200_iofpga {
public:
  c7200_iofpga(std::span<const std::uint16_t> cpu = {},
               std::span<const std::uint16_t> midplane = {},
               c7200_environment *environment = nullptr) noexcept
      : cpu_eeprom_(cpu), midplane_eeprom_(midplane),
        environment_(environment ? environment : &owned_environment_),
        endpoint_{this, read, write} {}

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
             nullptr,
             [](void *context, jit::devices::interrupt_line line) noexcept {
               let &self = *static_cast<c7200_iofpga *>(context);
               self.interrupt_.update(false);
               self.interrupt_ = line;
               self.update_interrupt();
             }}};
  }
  fn reset() noexcept -> void {
    cpu_eeprom_.reset();
    midplane_eeprom_.reset();
    eeprom_pins_ = 0;
    for (let &sensor : sensors_)
      sensor.reset();
    temperature_control_ = temperature_data_ = 0;
    console_.reset();
    auxiliary_.reset();
    control_ = 0;
    interrupt_mask_ = 0;
    tx_enabled_ = rx_enabled_ = {};
    update_interrupt();
  }

  template <typename Sink> fn drain(bool auxiliary, Sink &&sink) noexcept {
    let result =
        (auxiliary ? auxiliary_ : console_).drain(std::forward<Sink>(sink));
    update_interrupt();
    return result;
  }
  fn receive(std::span<const std::byte> bytes) noexcept -> std::size_t {
    if (!rx_enabled_[0])
      return 0;
    const let count = console_.receive(bytes);
    update_interrupt();
    return count;
  }

private:
  jit::uart console_{{0}}, auxiliary_{{0}};
  std::uint32_t control_ = 0;
  std::uint8_t interrupt_mask_ = 0;
  nmc93c46 cpu_eeprom_, midplane_eeprom_;
  std::uint32_t eeprom_pins_ = 0;
  std::array<ds1620, 4> sensors_{};
  std::uint32_t temperature_control_ = 0, temperature_data_ = 0;
  std::array<bool, 2> tx_enabled_{}, rx_enabled_{};
  jit::devices::interrupt_line interrupt_{};
  c7200_environment owned_environment_{};
  c7200_environment *environment_;
  jit::memory::device endpoint_;

  fn interrupt_status() const noexcept -> std::uint32_t {
    return ((tx_enabled_[0] && console_.pending() < jit::uart::capacity) ? 1u
                                                                         : 0u) |
           ((console_.pending_input()) ? 2u : 0u) |
           ((tx_enabled_[1] && auxiliary_.pending() < jit::uart::capacity)
                ? 0x10u
                : 0u) |
           ((auxiliary_.pending_input()) ? 0x20u : 0u);
  }
  fn update_interrupt() noexcept -> void {
    interrupt_.update((interrupt_status() & interrupt_mask_) != 0);
  }
  fn command(unsigned channel, std::uint8_t value) noexcept
      -> std::expected<void, jit::memory::device_error> {
    let &uart = channel ? auxiliary_ : console_;
    if (((value >> 4) & 7) >= 6)
      return std::unexpected(jit::memory::device_error::bus);
    if (value & 1)
      rx_enabled_[channel] = true;
    if (value & 2)
      rx_enabled_[channel] = false;
    if (value & 4)
      tx_enabled_[channel] = true;
    if (value & 8)
      tx_enabled_[channel] = false;
    if (((value >> 4) & 7) == 2) {
      rx_enabled_[channel] = false;
      uart.clear_input();
    }
    if (((value >> 4) & 7) == 3) {
      tx_enabled_[channel] = false;
      uart.clear_output();
    }
    update_interrupt();
    return {};
  }

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
    case 0x257: {
      if (access.size != jit::memory::access_size::byte)
        return std::unexpected(jit::memory::device_error::unsupported_size);
      const let sample = self.environment_->sample(self.temperature_data_);
      if (!sample)
        return std::unexpected(jit::memory::device_error::bus);
      return *sample;
    }
    case 0x20c:
      return self.temperature_control_;
    case 0x214:
      return self.temperature_data_;
    case 0x22c: {
      std::uint32_t value = 0;
      for (unsigned i = 0; i < self.sensors_.size(); i++)
        value |= static_cast<std::uint32_t>(self.sensors_[i].output()) << i;
      return value;
    }
    case 0x21c: {
      let value = self.eeprom_pins_;
      if (value & 4)
        value = (value & ~64u) | (self.cpu_eeprom_.output() ? 64u : 0u);
      if (value & 32)
        value = (value & ~128u) | (self.midplane_eeprom_.output() ? 128u : 0u);
      return value;
    }
    case 0x204:
      return self.control_ | 4u;
    case 0x224:
    case 0x234:
    case 0x244:
    case 0x398:
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
      const unsigned channel = offset == 0x40c ? 0u : 1u;
      return (((*status & 1) && self.tx_enabled_[channel]) ? 4u : 0u) |
             ((*status & 2) ? 1u : 0u);
    }
    case 0x41c:
    case 0x45c: {
      let &uart =
          (offset == 0x41c ? self.console_ : self.auxiliary_).endpoint();
      let result = uart.read(uart.context, uart_access(8));
      self.update_interrupt();
      return result;
    }
    case 0x42c:
      return self.interrupt_status();
    case 0x404:
    case 0x424:
    case 0x434:
    case 0x43c:
    case 0x444:
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
    case 0x20c:
      self.temperature_control_ = static_cast<std::uint32_t>(value) & 0x1f;
      for (unsigned i = 0; i < self.sensors_.size(); i++)
        if (!self.sensors_[i].pins((value & (1u << i)) != 0,
                                   (value & 0x10) != 0))
          return std::unexpected(jit::memory::device_error::bus);
      return {};
    case 0x214:
      self.temperature_data_ = static_cast<std::uint32_t>(value);
      for (let &sensor : self.sensors_)
        sensor.input((value & 1) != 0);
      return {};
    case 0x257:
    case 0x22c:
      return std::unexpected(jit::memory::device_error::bus);
    case 0x21c:
      self.eeprom_pins_ = static_cast<std::uint32_t>(value);
      self.cpu_eeprom_.write((value & 4) != 0, (value & 2) != 0,
                             (value & 1) != 0);
      self.midplane_eeprom_.write((value & 32) != 0, (value & 16) != 0,
                                  (value & 8) != 0);
      return {};
    case 0x204:
      self.control_ = static_cast<std::uint32_t>(value);
      return {};
    case 0x414:
    case 0x454:
      return self.command(offset == 0x414 ? 0u : 1u,
                          static_cast<std::uint8_t>(value));
    case 0x42c:
      self.interrupt_mask_ = static_cast<std::uint8_t>(value);
      self.update_interrupt();
      return {};
    case 0x41c:
    case 0x45c: {
      let &uart =
          (offset == 0x41c ? self.console_ : self.auxiliary_).endpoint();
      if (!self.tx_enabled_[offset == 0x41c ? 0u : 1u])
        return {};
      let result = uart.write(uart.context, uart_access(0), value);
      self.update_interrupt();
      return result;
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
