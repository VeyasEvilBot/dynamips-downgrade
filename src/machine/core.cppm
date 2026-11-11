module;

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include <dynamips/macros.h>

export module machine.core;

import host_memory;
import machine.firmware;
import jit.memory;
import jit.devices;
import jit.mips64;
import jit.dispatch;
import jit.execution;

export namespace dynamips::machine {
namespace mem = jit::memory;
inline constexpr std::uint64_t cached_base = 0xffffffff80000000ULL;
inline constexpr std::uint64_t uncached_base = 0xffffffffa0000000ULL;
inline constexpr std::uint64_t direct_size = 0x20000000;

struct cpu_definition {
  std::string_view name;
  std::uint32_t processor_id;
  jit::mips64::guest_address reset;
  jit::mips64::guest_address boot_stack{};
};

struct device_window {
  std::string_view name;
  jit::devices::physical_range range;
};

// Names and windows are borrowed immutable model data and outlive the instance.
struct definition {
  std::string_view name;
  cpu_definition cpu;
  std::size_t ram_bytes;
  jit::devices::physical_range rom;
  std::span<const device_window> windows;
};

struct window_id {
  std::uint8_t value;
};

struct attachment {
  window_id window;
  jit::devices::descriptor device;
};
enum class setup_issue {
  invalid_definition,
  invalid_attachment,
  invalid_firmware,
  boot_window_too_large
};

struct setup_error {
  std::variant<setup_issue, std::error_code, mem::mapping_error,
               jit::devices::registration_error, jit::dispatch_error>
      cause;
  std::size_t item = 0;
};

constexpr fn physical(std::uint64_t address) noexcept
    -> std::optional<std::uint64_t> {
  if (address < cached_base || address - cached_base >= 2 * direct_size)
    return std::nullopt;
  return (address - cached_base) & (direct_size - 1);
}

class instance {
public:
  instance(const instance &) = delete;
  fn operator=(const instance &) -> instance & = delete;

  [[nodiscard]] static fn
  create(definition model, const firmware::image &rom,
         const firmware::image *ios = nullptr,
         std::span<const attachment> attachments = {}) noexcept
      -> std::expected<std::unique_ptr<instance>, setup_error> {
    if (model.ram_bytes == 0 || model.ram_bytes % mem::page_size ||
        model.rom.size == 0 || model.rom.size % mem::page_size ||
        model.rom.base.value % mem::page_size ||
        model.rom.base.value < model.ram_bytes ||
        model.rom.base.value >= direct_size ||
        model.rom.size > direct_size - model.rom.base.value ||
        model.cpu.reset.value % 4 || !physical(model.cpu.reset.value) ||
        *physical(model.cpu.reset.value) < model.rom.base.value ||
        *physical(model.cpu.reset.value) - model.rom.base.value >=
            model.rom.size ||
        model.windows.size() > jit::devices::bus::capacity)
      return std::unexpected(setup_error{setup_issue::invalid_definition});
    for (std::size_t i = 0; i < model.windows.size(); i++) {
      const let range = model.windows[i].range;
      if (range.size == 0 || range.size % mem::page_size ||
          range.base.value % mem::page_size ||
          range.base.value < model.ram_bytes ||
          range.base.value >= direct_size ||
          range.size > direct_size - range.base.value ||
          (range.base.value < model.rom.base.value + model.rom.size &&
           model.rom.base.value < range.base.value + range.size))
        return std::unexpected(setup_error{setup_issue::invalid_definition, i});
      for (std::size_t j = 0; j < i; j++) {
        const let other = model.windows[j].range;
        if (range.base.value < other.base.value + other.size &&
            other.base.value < range.base.value + range.size)
          return std::unexpected(
              setup_error{setup_issue::invalid_definition, i});
      }
    }
    if (rom.entry() != model.cpu.reset.value)
      return std::unexpected(setup_error{setup_issue::invalid_firmware});
    try {
      let result = std::unique_ptr<instance>{new instance{model}};
      let ram = host::allocate_aligned(mem::page_size, model.ram_bytes);
      let backing = host::allocate_aligned(
          mem::page_size, static_cast<std::size_t>(model.rom.size));
      if (!ram)
        return std::unexpected(setup_error{ram.error()});
      if (!backing)
        return std::unexpected(setup_error{backing.error()});
      result->ram_ = std::move(*ram);
      result->rom_ = std::move(*backing);
      std::memset(result->ram_.get(), 0, model.ram_bytes);
      std::memset(result->rom_.get(), 0,
                  static_cast<std::size_t>(model.rom.size));
      const let rom_loaded = result->load(rom, false);
      if (!rom_loaded)
        return std::unexpected(rom_loaded.error());
      if (ios) {
        const let loaded = result->load(*ios, true);
        if (!loaded)
          return std::unexpected(loaded.error());
        result->ios_entry_ = ios->entry();
      }
      std::size_t pages =
          (model.ram_bytes + static_cast<std::size_t>(model.rom.size)) /
          mem::page_size;
      std::uint32_t attached = 0;
      for (std::size_t i = 0; i < attachments.size(); i++) {
        const let &entry = attachments[i];
        if (entry.window.value >= model.windows.size() ||
            (attached & (1u << entry.window.value)))
          return std::unexpected(
              setup_error{setup_issue::invalid_attachment, i});
        attached |= 1u << entry.window.value;
        const let range = model.windows[entry.window.value].range;
        const let registered = result->devices_.add(range, entry.device);
        if (!registered)
          return std::unexpected(setup_error{registered.error(), i});
        pages += static_cast<std::size_t>(range.size / mem::page_size);
      }
      let space =
          mem::address_space::create(std::as_writable_bytes(std::span{
                                         result->ram_.get(), model.ram_bytes}),
                                     pages * 2);
      if (!space)
        return std::unexpected(setup_error{space.error()});
      result->memory_ = std::move(*space);
      for (const std::uint64_t alias : std::array{cached_base, uncached_base}) {
        for (std::size_t offset = 0; offset < model.ram_bytes;
             offset += mem::page_size) {
          const let mapped = result->memory_->map_ram(
              {alias + offset}, {offset},
              mem::permission::read | mem::permission::write);
          if (!mapped)
            return std::unexpected(setup_error{mapped.error(), offset});
        }
        for (std::size_t offset = 0; offset < model.rom.size;
             offset += mem::page_size) {
          const let mapped = result->memory_->map_rom(
              {alias + model.rom.base.value + offset},
              {model.rom.base.value + offset},
              std::as_writable_bytes(
                  std::span{result->rom_.get() + offset, mem::page_size}));
          if (!mapped)
            return std::unexpected(setup_error{mapped.error(), offset});
        }
        for (const let &entry : attachments) {
          const let range = model.windows[entry.window.value].range;
          for (std::uint64_t offset = 0; offset < range.size;
               offset += mem::page_size) {
            const let mapped = result->memory_->map_device(
                {alias + range.base.value + offset},
                {range.base.value + offset}, result->devices_.endpoint(),
                mem::permission::read | mem::permission::write);
            if (!mapped)
              return std::unexpected(
                  setup_error{mapped.error(), entry.window.value});
          }
        }
      }
      const let start = *physical(model.cpu.reset.value) - model.rom.base.value;
      std::uint64_t end = start;
      for (const let &section : rom.sections()) {
        const let address = physical(section.address);
        if (address && *address >= model.rom.base.value &&
            *address - model.rom.base.value < model.rom.size)
          end = std::max(end, *address - model.rom.base.value + section.size);
      }
      if (end <= start || (end - start + 3) / 4 > 65536)
        return std::unexpected(setup_error{setup_issue::boot_window_too_large});
      result->boot_.resize(static_cast<std::size_t>((end - start + 3) / 4));
      for (std::size_t i = 0; i < result->boot_.size(); i++) {
        std::uint32_t word;
        std::memcpy(&word, result->rom_.get() + start + i * 4, sizeof(word));
        if constexpr (std::endian::native == std::endian::little)
          word = std::byteswap(word);
        result->boot_[i] = {word};
      }
      let dispatch = jit::dispatcher::create(result->boot_, model.cpu.reset,
                                              *result->memory_);
      if (!dispatch)
        return std::unexpected(setup_error{dispatch.error()});
      result->dispatch_ = std::move(*dispatch);
      result->reset();
      return result;
    } catch (const std::bad_alloc &) {
      return std::unexpected(
          setup_error{std::make_error_code(std::errc::not_enough_memory)});
    }
  }

  // Warm reset preserves RAM and loaded firmware; the owner calls this while
  // stopped.
  fn reset() noexcept -> void {
    state_ = {};
    state_.pc = model_.cpu.reset;
    state_.registers[29] = model_.cpu.boot_stack.value;
    devices_.reset();
  }
  fn model() const noexcept -> const definition & { return model_; }
  fn cpu() const noexcept -> const jit::mips64::state & { return state_; }
  fn memory() noexcept -> mem::address_space & { return *memory_; }
  fn ios_entry() const noexcept -> std::optional<std::uint64_t> {
    return ios_entry_;
  }

  [[nodiscard]] fn run_slice(std::uint64_t budget, std::size_t blocks = 64,
                              bool dump_ir = false) noexcept
      -> std::expected<jit::dispatch_result, jit::execution_error> {
    jit::execution_loop loop{*dispatch_, devices_};
    return loop.run_slice(state_, budget, blocks, dump_ir);
  }

private:
  definition model_;
  host::owned_bytes ram_, rom_;
  jit::devices::bus devices_;
  std::unique_ptr<mem::address_space> memory_;
  std::vector<jit::mips64::instruction_word> boot_;
  std::unique_ptr<jit::dispatcher> dispatch_;
  jit::mips64::state state_{};
  std::optional<std::uint64_t> ios_entry_;

  explicit instance(definition model) noexcept : model_(model) {}

  fn load(const firmware::image &image, bool ram_only) noexcept
      -> std::expected<void, setup_error> {
    for (std::size_t i = 0; i < image.sections().size(); i++) {
      const let &section = image.sections()[i];
      const let address = physical(section.address);
      if (!address || section.size > direct_size - *address)
        return std::unexpected(setup_error{setup_issue::invalid_firmware, i});
      for (std::size_t j = 0; j < i; j++) {
        const let &other = image.sections()[j];
        const let other_address = physical(other.address);
        if (other_address && *address < *other_address + other.size &&
            *other_address < *address + section.size)
          return std::unexpected(setup_error{setup_issue::invalid_firmware, i});
      }
      unsigned char *destination = nullptr;
      if (*address < model_.ram_bytes &&
          section.size <= model_.ram_bytes - *address)
        destination = ram_.get() + *address;
      else if (!ram_only && *address >= model_.rom.base.value &&
               *address - model_.rom.base.value < model_.rom.size &&
               section.size <=
                   model_.rom.size - (*address - model_.rom.base.value))
        destination = rom_.get() + (*address - model_.rom.base.value);
      if (!destination)
        return std::unexpected(setup_error{setup_issue::invalid_firmware, i});
      std::memset(destination, 0, static_cast<std::size_t>(section.size));
      if (!section.contents.empty())
        std::memcpy(destination, section.contents.data(),
                    section.contents.size());
    }
    return {};
  }
};
} // namespace dynamips::machine
