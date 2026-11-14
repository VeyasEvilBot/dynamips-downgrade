module;
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <dynamips/macros.h>
export module devices.gt64120;
import jit.devices;
import jit.memory;

export namespace dynamips::devices {
enum class bridge_model : std::uint16_t {
  dec21050 = 0x0001,
  dec21150 = 0x0023
};

struct pci_bridge_definition {
  std::uint8_t root;
  std::uint8_t device;
  bridge_model model;
};
enum class bridge_error { capacity, invalid_definition, duplicate_device };

struct pci_function {
  std::uint8_t root, bus, device, function;
  void *context;
  fn (*read)(void *, std::uint8_t) noexcept -> std::uint32_t;
  fn (*write)(void *, std::uint8_t, std::uint32_t) noexcept -> void;
};
enum class pci_function_error {
  capacity,
  invalid_definition,
  duplicate_device
};

// Startup registers and PCI configuration. DMA engines are not implemented.
class gt64120 {
public:
  explicit gt64120(jit::memory::physical_address base) noexcept
      : base_(base), endpoint_{this, read, write} {}

  gt64120(const gt64120 &) = delete;
  fn operator=(const gt64120 &) -> gt64120 & = delete;
  fn reset() noexcept -> void {
    bridge_bus_ = {};
    pci_address_ = {};
    interrupt_mask_ = 0;
  }

  // Configure while stopped; borrowed definitions outlive the controller.
  [[nodiscard]] fn
  configure_bridges(std::span<const pci_bridge_definition> definitions) noexcept
      -> std::expected<void, bridge_error> {
    if (definitions.size() > bridge_bus_.size())
      return std::unexpected(bridge_error::capacity);
    for (std::size_t i = 0; i < definitions.size(); i++) {
      const let &value = definitions[i];
      if (value.root > 1 || value.device == 0 || value.device > 31 ||
          (value.model != bridge_model::dec21050 &&
           value.model != bridge_model::dec21150))
        return std::unexpected(bridge_error::invalid_definition);
      for (std::size_t j = 0; j < function_count_; j++) {
        const let &function = functions_[j];
        if (function.root == value.root && function.bus == 0 &&
            function.device == value.device && function.function == 0)
          return std::unexpected(bridge_error::duplicate_device);
      }
      for (std::size_t j = 0; j < i; j++)
        if (definitions[j].root == value.root &&
            definitions[j].device == value.device)
          return std::unexpected(bridge_error::duplicate_device);
    }
    bridges_ = definitions;
    bridge_bus_ = {};
    return {};
  }

  [[nodiscard]] fn
  configure_functions(std::span<const pci_function> definitions) noexcept
      -> std::expected<void, pci_function_error> {
    if (definitions.size() > functions_.size())
      return std::unexpected(pci_function_error::capacity);
    for (std::size_t i = 0; i < definitions.size(); i++) {
      const let &value = definitions[i];
      if (value.root > 1 || value.device > 31 || value.function > 7 ||
          !value.context || !value.read || !value.write ||
          (value.bus == 0 && value.device == 0))
        return std::unexpected(pci_function_error::invalid_definition);
      for (const let &bridge : bridges_)
        if (value.bus == 0 && value.function == 0 &&
            value.root == bridge.root && value.device == bridge.device)
          return std::unexpected(pci_function_error::duplicate_device);
      for (std::size_t j = 0; j < i; j++) {
        const let &other = definitions[j];
        if (value.root == other.root && value.bus == other.bus &&
            value.device == other.device && value.function == other.function)
          return std::unexpected(pci_function_error::duplicate_device);
      }
    }
    for (std::size_t i = 0; i < definitions.size(); i++)
      functions_[i] = definitions[i];
    function_count_ = definitions.size();
    return {};
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
  std::span<const pci_bridge_definition> bridges_;
  std::array<std::uint32_t, 8> bridge_bus_{};
  std::array<pci_function, 8> functions_{};
  std::size_t function_count_ = 0;
  jit::memory::device endpoint_;

  fn pci_read(unsigned root) const noexcept -> std::uint32_t {
    const let address = pci_address_[root];
    const let reg = address & 0xfc;
    for (std::size_t i = 0; i < function_count_; i++) {
      const let &entry = functions_[i];
      if (entry.root == root && entry.bus == ((address >> 16) & 255) &&
          entry.device == ((address >> 11) & 31) &&
          entry.function == ((address >> 8) & 7))
        return entry.read(entry.context, static_cast<std::uint8_t>(reg));
    }
    const let device = (address >> 11) & 31;
    if ((address & 0x00ff0700) != 0)
      return reg == 0 ? 0xffffffffu : 0;
    if (root == 0 && device == 0) {
      if (reg == 0)
        return 0x462011ab;
      if (reg == 8)
        return 0x03008005;
      return 0;
    }
    for (std::size_t i = 0; i < bridges_.size(); i++) {
      const let &bridge = bridges_[i];
      if (bridge.root != root || bridge.device != device)
        continue;
      if (reg == 0)
        return (static_cast<std::uint32_t>(bridge.model) << 16) | 0x1011u;
      if (reg == 8)
        return 0x06040000;
      if (reg == 0x18)
        return bridge_bus_[i];
      return 0;
    }
    return reg == 0 ? 0xffffffffu : 0;
  }
  fn pci_write(unsigned root, std::uint32_t value) noexcept -> void {
    const let address = pci_address_[root];
    for (std::size_t i = 0; i < function_count_; i++) {
      const let &entry = functions_[i];
      if (entry.root == root && entry.bus == ((address >> 16) & 255) &&
          entry.device == ((address >> 11) & 31) &&
          entry.function == ((address >> 8) & 7)) {
        entry.write(entry.context, static_cast<std::uint8_t>(address & 0xfc),
                    value);
        return;
      }
    }
    if ((address & 0x00ff07ff) != 0x18)
      return;
    const let device = (address >> 11) & 31;
    for (std::size_t i = 0; i < bridges_.size(); i++)
      if (bridges_[i].root == root && bridges_[i].device == device) {
        bridge_bus_[i] = value;
        return;
      }
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
      self.pci_write(1, static_cast<std::uint32_t>(value));
      return {};
    case 0xcfc:
      self.pci_write(0, static_cast<std::uint32_t>(value));
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
