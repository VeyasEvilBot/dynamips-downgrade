module;
#include <array>
#include <bit>
#include <cstdint>
#include <expected>
#include <dynamips/macros.h>
export module devices.c7200_midplane;
import jit.devices;
import jit.memory;
import devices.c7200_environment;

export namespace dynamips::devices {
// Empty port-adapter bays. Unpopulated serial EEPROM data lines are pulled
// high.
class c7200_midplane {
public:
  explicit c7200_midplane(c7200_environment *environment = nullptr) noexcept
      : environment_(environment ? environment : &owned_environment_),
        endpoint_{this, read, write} {}

  c7200_midplane(const c7200_midplane &) = delete;
  fn operator=(const c7200_midplane &) -> c7200_midplane & = delete;

  fn descriptor() noexcept -> jit::devices::descriptor {
    using jit::devices::widths;
    return {&endpoint_,
            widths::byte | widths::half | widths::word,
            std::endian::big,
            {this,
             [](void *context) noexcept {
               static_cast<c7200_midplane *>(context)->reset();
             },
             nullptr}};
  }
  fn reset() noexcept -> void {
    masks_ = {};
    control_ = 0;
    pins_ = {0, 3};
    environment_->power_enabled = 3;
  }

private:
  std::array<std::uint32_t, 2> masks_{};
  std::uint32_t control_ = 0;
  std::array<std::uint32_t, 2> pins_{0, 3};
  c7200_environment owned_environment_{};
  c7200_environment *environment_;
  jit::memory::device endpoint_;

  static fn valid_size(jit::memory::access_size size) noexcept -> bool {
    using jit::memory::access_size;
    return size == access_size::byte || size == access_size::half ||
           size == access_size::word;
  }

  static fn absent_eeproms(std::uint32_t pins, unsigned first) noexcept
      -> std::uint32_t {
    for (unsigned shift = first; shift < 32; shift += 8)
      if (pins & (2u << shift))
        pins |= 64u << shift;
    return pins;
  }

  static fn read(void *context, jit::memory::device_access access) noexcept
      -> std::expected<std::uint64_t, jit::memory::device_error> {
    if (!valid_size(access.size))
      return std::unexpected(jit::memory::device_error::unsupported_size);
    let &self = *static_cast<c7200_midplane *>(context);
    switch (access.address.value - 0x1e800000) {
    case 0x10:
    case 0x11:
    case 0x12:
    case 0x13:
    case 0x18:
    case 0x19:
    case 0x1a:
    case 0x1b:
    case 0x38:
    case 0x40:
    case 0x70:
    case 0x78:
    case 0x7b:
      return 0;
    case 0x20:
      return self.masks_[0];
    case 0x28:
      return self.masks_[1];
    case 0x48:
      return 0xffffffffu;
    case 0x50:
      return 6;
    case 0x58:
      return self.control_;
    case 0x60:
      return absent_eeproms(self.pins_[0], 0);
    case 0x68:
      return absent_eeproms(self.pins_[1], 8);
    default:
      return std::unexpected(jit::memory::device_error::bus);
    }
  }

  static fn write(void *context, jit::memory::device_access access,
                   std::uint64_t value) noexcept
      -> std::expected<void, jit::memory::device_error> {
    if (!valid_size(access.size))
      return std::unexpected(jit::memory::device_error::unsupported_size);
    let &self = *static_cast<c7200_midplane *>(context);
    const let word = static_cast<std::uint32_t>(value);
    switch (access.address.value - 0x1e800000) {
    case 0x20:
      self.masks_[0] = word;
      return {};
    case 0x28:
      self.masks_[1] = word;
      return {};
    case 0x58:
      self.control_ = word;
      return {};
    case 0x60:
      self.pins_[0] = word;
      return {};
    case 0x68:
      if ((word & 3) == 0)
        return std::unexpected(jit::memory::device_error::bus);
      self.pins_[1] = word;
      self.environment_->power_enabled = static_cast<std::uint8_t>(word & 3);
      return {};
    default:
      const let known = read(context, access);
      if (!known)
        return std::unexpected(known.error());
      return {};
    }
  }
};
} // namespace dynamips::devices
