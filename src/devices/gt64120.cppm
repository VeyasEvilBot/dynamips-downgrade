module;
#include <array>
#include <bit>
#include <cstdint>
#include <expected>
#include <dynamips/macros.h>
export module devices.gt64120;
import jit.devices;
import jit.memory;

export namespace dynamips::devices {
// Startup registers and empty PCI buses; DMA engines are not implemented.
class gt64120 {
public:
  explicit gt64120(jit::memory::physical_address base) noexcept
      : base_(base), endpoint_{this, read, write} {}

  gt64120(const gt64120 &) = delete;
  fn operator=(const gt64120 &) -> gt64120 & = delete;
  fn reset() noexcept -> void {
    pci_address_ = {};
    interrupt_mask_ = 0;
  }
  fn descriptor() noexcept -> jit::devices::descriptor {
    return {&endpoint_,
            jit::devices::widths::word,
            std::endian::little,
            {this,
             [](void *context) noexcept {
               static_cast<gt64120 *>(context)->reset();
             },
             nullptr}};
  }

private:
  jit::memory::physical_address base_;
  std::array<std::uint32_t, 2> pci_address_{};
  std::uint32_t interrupt_mask_ = 0;
  jit::memory::device endpoint_;

  fn pci_read(unsigned bus) const noexcept -> std::uint32_t {
    const let address = pci_address_[bus];
    const let reg = address & 255;
    if (bus != 0 || (address & 0x00ffff00) != 0)
      return reg == 0 ? 0xffffffffu : 0;
    if (reg == 0)
      return 0x462011ab;
    if (reg == 8)
      return 0x03008005;
    return 0;
  }

  static fn read(void *context, jit::memory::device_access access) noexcept
      -> std::expected<std::uint64_t, jit::memory::device_error> {
    if (access.size != jit::memory::access_size::word)
      return std::unexpected(jit::memory::device_error::unsupported_size);
    let &self = *static_cast<gt64120 *>(context);
    switch (access.address.value - self.base_.value) {
    // Decode-window setup retains the legacy read-zero/write-ignore behavior.
    case 0x048:
    case 0x050:
    case 0x058:
    case 0x060:
    case 0x080:
    case 0x088:
    case 0x090:
    case 0x098:
    case 0x0a0:
    case 0x0a8:
    case 0x0b0:
    case 0x0b8:
    case 0xc04:
    case 0xc10:
    case 0xc14:
    case 0xc24:
    case 0xc3c:
    case 0xc40:
    case 0xc48:
    case 0xc58:
    case 0xc80:
    case 0xc84:
    case 0xc88:
    case 0xc8c:
    case 0xc90:
    case 0xc94:
    case 0xca4:
    case 0xcbc:
    case 0xcc0:
    case 0xcc8:
    case 0xcd8:
    case 0x000:
    case 0x864:
    case 0x008:
    case 0x400:
    case 0x40c:
    case 0x410:
    case 0x41c:
    case 0xc18:
    case 0xc98:
    case 0xc9c:
      return 0;
    case 0x010:
    case 0x020:
    case 0x408:
    case 0x418:
      return 0x7f;
    case 0x018:
      return 0x100;
    case 0x404:
    case 0x414:
      return 0xff;
    case 0xc08:
    case 0xc0c:
      return 0xfff;
    case 0xc00:
      return 0x8001;
    case 0xc1c:
      return self.interrupt_mask_;
    case 0xcf0:
      return self.pci_address_[1];
    case 0xcf4:
      return self.pci_read(1);
    case 0xcf8:
      return self.pci_address_[0];
    case 0xcfc:
      return self.pci_read(0);
    default:
      return std::unexpected(jit::memory::device_error::bus);
    }
  }

  static fn write(void *context, jit::memory::device_access access,
                   std::uint64_t value) noexcept
      -> std::expected<void, jit::memory::device_error> {
    if (access.size != jit::memory::access_size::word)
      return std::unexpected(jit::memory::device_error::unsupported_size);
    let &self = *static_cast<gt64120 *>(context);
    switch (access.address.value - self.base_.value) {
    case 0x864:
      if (value != 0)
        return std::unexpected(jit::memory::device_error::bus);
      return {};
    case 0xc1c:
      self.interrupt_mask_ = static_cast<std::uint32_t>(value);
      return {};
    case 0xcf0:
      self.pci_address_[1] = static_cast<std::uint32_t>(value);
      return {};
    case 0xcf8:
      self.pci_address_[0] = static_cast<std::uint32_t>(value);
      return {};
    case 0xcf4:
    case 0xcfc:
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
