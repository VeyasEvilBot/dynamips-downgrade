module;

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <utility>

#include <dynamips/macros.h>

export module devices.rom_services;

import jit.devices;
import jit.fifo;
import jit.memory;

export namespace dynamips::devices {
struct rom_boot_info {
  jit::memory::virtual_address nvram;
  jit::memory::virtual_address ios_entry{};
  std::uint16_t elf_machine_id = 0;
};

enum class rom_output : std::uint8_t { console, log };

// Console +0x40 and ROM log +0x3c accept the low byte of big-endian words.
// Other ROM services remain unsupported. The owner drains between slices.
class rom_services {
public:
  static constexpr std::size_t capacity = 1024;

  explicit rom_services(jit::memory::physical_address base,
                        rom_boot_info boot = {}) noexcept
      : base_(base), boot_(boot), endpoint_{this, read, write} {}

  rom_services(const rom_services &) = delete;
  fn operator=(const rom_services &) -> rom_services & = delete;

  fn descriptor() noexcept -> jit::devices::descriptor {
    return {&endpoint_,
            jit::devices::widths::word,
            std::endian::big,
            {this,
             [](void *context) noexcept {
               static_cast<rom_services *>(context)->reset();
             },
             nullptr}};
  }

  fn reset() noexcept -> void { queues_ = {}; }

  template <typename Sink> fn drain(rom_output channel, Sink &&sink) noexcept {
    return queue(channel).drain(std::forward<Sink>(sink));
  }

private:
  jit::memory::physical_address base_;
  rom_boot_info boot_;
  std::array<jit::byte_fifo<capacity>, 2> queues_{};
  jit::memory::device endpoint_;

  fn queue(rom_output channel) noexcept -> jit::byte_fifo<capacity> & {
    return channel == rom_output::console ? queues_[0] : queues_[1];
  }

  static fn read(void *context, jit::memory::device_access access) noexcept
      -> std::expected<std::uint64_t, jit::memory::device_error> {
    namespace mem = jit::memory;
    let &self = *static_cast<rom_services *>(context);
    if (access.size != mem::access_size::word)
      return std::unexpected(mem::device_error::unsupported_size);
    if (access.address.value < self.base_.value)
      return std::unexpected(mem::device_error::bus);
    std::uint32_t value;
    switch (access.address.value - self.base_.value) {
    case 0:
      value = 0x1e94b3df;
      break;
    case 0x28:
      value = static_cast<std::uint32_t>(self.boot_.ios_entry.value);
      break;
    case 0x2c:
      value = self.boot_.elf_machine_id;
      break;
    case 0x44:
      value = static_cast<std::uint32_t>(self.boot_.nvram.value);
      break;
    default:
      return std::unexpected(mem::device_error::bus);
    }
    return access.order == std::endian::little ? std::byteswap(value) : value;
  }

  static fn write(void *context, jit::memory::device_access access,
                   std::uint64_t value) noexcept
      -> std::expected<void, jit::memory::device_error> {
    namespace mem = jit::memory;
    let &self = *static_cast<rom_services *>(context);
    if (access.size != mem::access_size::word)
      return std::unexpected(mem::device_error::unsupported_size);
    if (access.address.value < self.base_.value)
      return std::unexpected(mem::device_error::bus);
    const let offset = access.address.value - self.base_.value;
    if (offset != 0x40 && offset != 0x3c)
      return std::unexpected(mem::device_error::bus);
    let word = static_cast<std::uint32_t>(value);
    if (access.order == std::endian::little)
      word = std::byteswap(word);
    const let channel = offset == 0x40 ? rom_output::console : rom_output::log;
    if (!self.queue(channel).push(static_cast<std::byte>(word & 0xff)))
      return std::unexpected(mem::device_error::bus);
    return {};
  }
};
} // namespace dynamips::devices
