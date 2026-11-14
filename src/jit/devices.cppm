module;

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <system_error>

#include <dynamips/macros.h>

export module jit.devices;

import jit.memory;

export namespace jit::devices {
struct physical_range {
  memory::physical_address base;
  std::uint64_t size;
};

enum class widths : std::uint8_t {
  byte = 1,
  half = 2,
  word = 4,
  doubleword = 8
};

constexpr fn operator|(widths left, widths right) noexcept -> widths {
  return static_cast<widths>(static_cast<unsigned>(left) |
                             static_cast<unsigned>(right));
}

struct interrupt_line {
  void *context = nullptr;
  fn (*set_level)(void *, bool) noexcept -> void = nullptr;
  fn update(bool level) const noexcept -> void {
    if (set_level)
      set_level(context, level);
  }
};

struct hooks {
  void *context = nullptr;
  fn (*reset)(void *) noexcept -> void = nullptr;
  fn (*service)(void *) noexcept -> std::expected<void, std::error_code> = nullptr;
  fn (*connect_interrupt)(void *, interrupt_line) noexcept -> void = nullptr;
};

// The borrowed endpoint returns numeric registers in register_order. The bus
// converts to the guest access order. Hooks run only while guest execution
// stops.
struct descriptor {
  memory::device *endpoint;
  widths supported;
  std::endian register_order;
  hooks lifecycle{};
};

struct device_id {
  std::uint8_t value;
};
enum class registration_error {
  invalid_range,
  invalid_descriptor,
  overlap,
  full,
  sealed
};

struct service_error {
  device_id device;
  std::error_code cause;
};

// One owner. Devices, hook contexts and this bus outlive mapped memory
// accesses. Registration finishes before endpoint() exposes the stable routing
// table.
class bus {
public:
  static constexpr std::size_t capacity = 16;

  bus() noexcept : endpoint_{this, read, write} {}

  bus(const bus &) = delete;
  fn operator=(const bus &) -> bus & = delete;

  [[nodiscard]] fn add(physical_range range, descriptor device) noexcept
      -> std::expected<device_id, registration_error> {
    if (sealed_)
      return std::unexpected(registration_error::sealed);
    if (range.size == 0 || range.size - 1 > UINT64_MAX - range.base.value)
      return std::unexpected(registration_error::invalid_range);
    const let mask = static_cast<unsigned>(device.supported);
    if (!device.endpoint ||
        (!device.endpoint->read && !device.endpoint->write) || mask == 0 ||
        (mask & ~15u) != 0 ||
        (device.register_order != std::endian::big &&
         device.register_order != std::endian::little))
      return std::unexpected(registration_error::invalid_descriptor);
    for (std::size_t i = 0; i < size_; i++) {
      const let other = entries_[i].range;
      if (range.base.value <= other.base.value + other.size - 1 &&
          other.base.value <= range.base.value + range.size - 1)
        return std::unexpected(registration_error::overlap);
    }
    if (size_ == capacity)
      return std::unexpected(registration_error::full);
    const device_id id{static_cast<std::uint8_t>(size_)};
    entries_[size_++] = {range, device};
    return id;
  }

  fn endpoint() noexcept -> memory::device & {
    sealed_ = true;
    return endpoint_;
  }

  fn reset() noexcept -> void {
    for (std::size_t i = 0; i < size_; i++) {
      const let &hook = entries_[i].device.lifecycle;
      if (hook.reset)
        hook.reset(hook.context);
    }
  }

  [[nodiscard]] fn service() noexcept -> std::expected<void, service_error> {
    for (std::size_t i = 0; i < size_; i++) {
      const let &hook = entries_[i].device.lifecycle;
      if (!hook.service)
        continue;
      if (const let result = hook.service(hook.context); !result)
        return std::unexpected(
            service_error{{static_cast<std::uint8_t>(i)}, result.error()});
    }
    return {};
  }

private:
  struct entry {
    physical_range range;
    descriptor device;
  };

  std::array<entry, capacity> entries_{};
  std::size_t size_ = 0;
  bool sealed_ = false;
  memory::device endpoint_;

  fn find(memory::device_access access) noexcept
      -> std::expected<descriptor *, memory::device_error> {
    const let width = static_cast<unsigned>(access.size);
    if (width != 1 && width != 2 && width != 4 && width != 8)
      return std::unexpected(memory::device_error::unsupported_size);
    for (std::size_t i = 0; i < size_; i++) {
      let &entry = entries_[i];
      if (access.address.value < entry.range.base.value)
        continue;
      const let offset = access.address.value - entry.range.base.value;
      if (offset >= entry.range.size)
        continue;
      if (width > entry.range.size - offset)
        return std::unexpected(memory::device_error::bus);
      if ((static_cast<unsigned>(entry.device.supported) & width) == 0)
        return std::unexpected(memory::device_error::unsupported_size);
      return &entry.device;
    }
    return std::unexpected(memory::device_error::bus);
  }

  static fn convert(std::uint64_t value, memory::access_size size) noexcept
      -> std::uint64_t {
    switch (size) {
    case memory::access_size::byte:
      return static_cast<std::uint8_t>(value);
    case memory::access_size::half:
      return std::byteswap(static_cast<std::uint16_t>(value));
    case memory::access_size::word:
      return std::byteswap(static_cast<std::uint32_t>(value));
    case memory::access_size::doubleword:
      return std::byteswap(value);
    }
    return value;
  }

  static fn read(void *context, memory::device_access access) noexcept
      -> std::expected<std::uint64_t, memory::device_error> {
    let &self = *static_cast<bus *>(context);
    const let found = self.find(access);
    if (!found)
      return std::unexpected(found.error());
    const let &device = **found;
    if (!device.endpoint->read)
      return std::unexpected(memory::device_error::bus);
    const let order = access.order;
    access.order = device.register_order;
    const let value = device.endpoint->read(device.endpoint->context, access);
    if (!value)
      return std::unexpected(value.error());
    return order == access.order ? *value : convert(*value, access.size);
  }

  static fn write(void *context, memory::device_access access,
                   std::uint64_t value) noexcept
      -> std::expected<void, memory::device_error> {
    let &self = *static_cast<bus *>(context);
    const let found = self.find(access);
    if (!found)
      return std::unexpected(found.error());
    const let &device = **found;
    if (!device.endpoint->write)
      return std::unexpected(memory::device_error::bus);
    if (access.order != device.register_order)
      value = convert(value, access.size);
    access.order = device.register_order;
    return device.endpoint->write(device.endpoint->context, access, value);
  }
};
} // namespace jit::devices
