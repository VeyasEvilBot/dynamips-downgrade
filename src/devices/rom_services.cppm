module;

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <utility>
#include <span>
#include <string_view>
#include <limits>

#include <dynamips/macros.h>

export module devices.rom_services;

import jit.devices;
import jit.fifo;
import jit.memory;

export namespace dynamips::devices {
struct rommon_variable {
  std::string_view name;
  std::string_view value;
};

struct rom_boot_info {
  jit::memory::virtual_address nvram;
  jit::memory::virtual_address ios_entry{};
  std::uint16_t elf_machine_id = 0;
  std::uint32_t ram_megabytes = 0;
  std::uint16_t configuration_register = 0;
  // Borrowed immutable boot environment; the owner outlives the device.
  std::span<const rommon_variable> variables{};
  std::uint32_t rom_megabytes = 0;
  std::uint32_t nvram_kibibytes = 0;
  std::uint32_t iomem_megabytes = 0;
};

enum class rom_output : std::uint8_t { console, log };

// Console +0x40 and ROM log +0x3c accept the low byte of big-endian words.
// Other ROM services remain unsupported. The owner drains between slices.
class rom_services {
public:
  static constexpr std::size_t capacity = 1024;
  static constexpr std::size_t variable_capacity = 64;

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

  fn reset() noexcept -> void {
    queues_ = {};
    variable_buffer_ = {};
    variable_position_ = 0;
    variable_status_ = 0;
    variable_overflow_ = false;
  }

  template <typename Sink> fn drain(rom_output channel, Sink &&sink) noexcept {
    return queue(channel).drain(std::forward<Sink>(sink));
  }

private:
  jit::memory::physical_address base_;
  rom_boot_info boot_;
  std::array<jit::byte_fifo<capacity>, 2> queues_{};
  jit::memory::device endpoint_;
  std::array<char, 512> variable_buffer_{};
  std::size_t variable_position_ = 0;
  std::uint32_t variable_status_ = 0;
  bool variable_overflow_ = false;

  struct stored_variable {
    std::array<char, 512> text{};
    std::size_t name_length = 0;
    bool used = false;
  };

  std::array<stored_variable, variable_capacity> variables_{};

  fn set_variable() noexcept -> void {
    const std::string_view assignment{variable_buffer_.data()};
    const let equals = assignment.find('=');
    variable_status_ = std::numeric_limits<std::uint32_t>::max();
    if (!variable_overflow_ && equals != std::string_view::npos &&
        equals != 0) {
      const let name = assignment.substr(0, equals);
      stored_variable *destination = nullptr;
      for (let &entry : variables_)
        if (entry.used &&
            std::string_view{entry.text.data(), entry.name_length} == name) {
          destination = &entry;
          break;
        }
      if (!destination)
        for (let &entry : variables_)
          if (!entry.used) {
            destination = &entry;
            break;
          }
      if (destination) {
        std::ranges::copy(assignment, destination->text.begin());
        destination->text[assignment.size()] = 0;
        destination->name_length = equals;
        destination->used = true;
        variable_buffer_[equals] = 0;
        variable_status_ = 0;
      }
    }
    variable_position_ = 0;
    variable_overflow_ = false;
  }

  fn query_variable() noexcept -> void {
    const std::string_view name{variable_buffer_.data()};
    if (!variable_overflow_)
      for (const let &entry : variables_)
        if (entry.used &&
            std::string_view{entry.text.data(), entry.name_length} == name) {
          const std::string_view value{entry.text.data() + entry.name_length +
                                       1};
          std::ranges::copy(value, variable_buffer_.begin());
          variable_buffer_[value.size()] = 0;
          variable_position_ = 0;
          variable_status_ = 0;
          return;
        }
    const let found =
        std::ranges::find(boot_.variables, name, &rommon_variable::name);
    const bool available = !variable_overflow_ &&
                           found != boot_.variables.end() &&
                           found->value.size() < variable_buffer_.size();
    variable_status_ =
        available ? 0 : std::numeric_limits<std::uint32_t>::max();
    if (available) {
      std::ranges::copy(found->value, variable_buffer_.begin());
      variable_buffer_[found->value.size()] = 0;
    } else
      variable_buffer_[0] = 0;
    variable_position_ = 0;
    variable_overflow_ = false;
  }

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
    case 0x14:
      value = self.boot_.ram_megabytes;
      break;
    case 0x18:
      value = self.boot_.rom_megabytes;
      break;
    case 0x1c:
      value = self.boot_.nvram_kibibytes;
      break;
    case 0x20:
      value = self.boot_.iomem_megabytes;
      break;
    case 0x24:
      value = self.boot_.configuration_register;
      break;
    case 0x28:
      value = static_cast<std::uint32_t>(self.boot_.ios_entry.value);
      break;
    case 0x2c:
      value = self.boot_.elf_machine_id;
      break;
    case 0x54:
      value = self.variable_position_ < self.variable_buffer_.size()
                  ? static_cast<unsigned char>(
                        self.variable_buffer_[self.variable_position_])
                  : 0;
      if (value != 0)
        self.variable_position_++;
      break;
    case 0x58:
      value = self.variable_status_;
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
    let word = static_cast<std::uint32_t>(value);
    if (access.order == std::endian::little)
      word = std::byteswap(word);
    if (offset == 0x54) {
      if (self.variable_position_ < self.variable_buffer_.size() - 1) {
        self.variable_buffer_[self.variable_position_++] =
            static_cast<char>(word & 255);
        self.variable_buffer_[self.variable_position_] = 0;
      } else if ((word & 255) != 0)
        self.variable_overflow_ = true;
      return {};
    }
    if (offset == 0x58) {
      switch (word & 255) {
      case 1:
        self.set_variable();
        return {};
      case 2:
        self.query_variable();
        return {};
      case 3:
        self.variable_position_ = 0;
        self.variable_overflow_ = false;
        return {};
      default:
        return std::unexpected(mem::device_error::bus);
      }
    }
    if (offset != 0x40 && offset != 0x3c)
      return std::unexpected(mem::device_error::bus);
    const let channel = offset == 0x40 ? rom_output::console : rom_output::log;
    if (!self.queue(channel).push(static_cast<std::byte>(word & 0xff)))
      return std::unexpected(mem::device_error::bus);
    return {};
  }
};
} // namespace dynamips::devices
