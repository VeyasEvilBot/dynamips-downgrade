module;

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <new>
#include <limits>
#include <span>
#include <charconv>
#include <chrono>
#include <memory>
#include <cstdint>
#include <exception>
#include <expected>
#include <utility>
#include <optional>
#include <print>
#include <string_view>
#include <system_error>
#include <variant>

#include <dynamips/macros.h>

export module boot.application;

import devices.byte_memory;
import devices.bootflash;
import host_memory;
import devices.c7200_midplane;
import devices.c7200_environment;
import devices.gt64120;
import devices.clpd6729;
import devices.c7200_iofpga;
import devices.rom_services;
import machine.firmware;
import machine.core;
import machine.c7200;
import jit.memory;
import jit.dispatch;
import jit.amd64;
import jit.devices;
import host.console;

namespace {
namespace machine = dynamips::machine;
fn load(const char *path)
    -> std::expected<dynamips::firmware::image, dynamips::firmware::error> {
  let result = dynamips::firmware::image::read(path);
  if (!result) {
    const let &error = result.error();
    if (error.host)
      std::println(stderr, "boot: {}: {}", path, error.host.message());
    else
      std::println(stderr, "boot: {}: {} at section {}", path,
                   dynamips::firmware::message(error.code), error.section);
  }
  return result;
}

fn setup_message(machine::setup_issue issue) noexcept -> std::string_view {
  switch (issue) {
  case machine::setup_issue::invalid_definition:
    return "invalid chassis memory or device layout";
  case machine::setup_issue::invalid_attachment:
    return "duplicate or invalid device attachment";
  case machine::setup_issue::invalid_firmware:
    return "firmware entry or section does not fit the chassis RAM/ROM layout";
  case machine::setup_issue::boot_window_too_large:
    return "ROM boot window exceeds the current dispatcher limit";
  }
  return "machine setup failed";
}

template <typename Drain>
fn flush_output(Drain &&drain,
                  dynamips::host::console_transport *transport = nullptr,
                  bool *blocked = nullptr)
    -> std::expected<void, std::error_code> {
  for (const bool auxiliary : {false, true}) {
    FILE *stream = !auxiliary ? stdout : stderr;
    const let drained =
        drain(auxiliary,
              [stream, auxiliary, transport,
               blocked](std::span<const std::byte> bytes) noexcept
                  -> std::expected<std::size_t, std::error_code> {
                if (auxiliary)
                  return bytes.size();
                if (transport) {
                  const let count = transport->send(bytes);
                  if (blocked && count != bytes.size())
                    *blocked = true;
                  return count;
                }
                try {
                  std::print(stream, "{}",
                             std::string_view{
                                 reinterpret_cast<const char *>(bytes.data()),
                                 bytes.size()});
                  return bytes.size();
                } catch (const std::system_error &error) {
                  return std::unexpected(error.code());
                } catch (const std::bad_alloc &) {
                  return std::unexpected(
                      std::make_error_code(std::errc::not_enough_memory));
                }
              });
    if (!drained) {
      if (const let *cause =
              std::get_if<std::error_code>(&drained.error().cause))
        return std::unexpected(*cause);
      return std::unexpected(std::make_error_code(std::errc::protocol_error));
    }
    if (!transport && !auxiliary && *drained != 0 && std::fflush(stream) != 0)
      return std::unexpected(std::error_code{errno, std::generic_category()});
  }
  return {};
}

fn flush_consoles(dynamips::devices::rom_services &services,
                    dynamips::devices::c7200_iofpga &io,
                    dynamips::host::console_transport *transport = nullptr,
                    bool *blocked = nullptr)
    -> std::expected<void, std::error_code> {
  const let rom = flush_output(
      [&services](bool auxiliary, let &&sink) {
        using dynamips::devices::rom_output;
        return services.drain(auxiliary ? rom_output::log : rom_output::console,
                              std::forward<decltype(sink)>(sink));
      },
      transport, blocked);
  if (!rom)
    return std::unexpected(rom.error());
  return flush_output(
      [&io](bool auxiliary, let &&sink) {
        return io.drain(auxiliary, std::forward<decltype(sink)>(sink));
      },
      transport, blocked);
}

// Service boundaries own all host I/O. Pending bytes stay in their source FIFO
// until the transport accepts them; slow readers suspend guest execution.
// Idle host polling is capped at 1 kHz; pending output bypasses that cadence.
fn service_console(dynamips::host::console_transport &transport,
                     dynamips::devices::rom_services &services,
                     dynamips::devices::c7200_iofpga &io,
                     std::chrono::steady_clock::time_point &next_poll)
    -> std::expected<bool, std::error_code> {
  for (;;) {
    const let now = std::chrono::steady_clock::now();
    if (now >= next_poll || transport.pending_output()) {
      const let polled = transport.poll(
          std::chrono::milliseconds{transport.pending_output() ? 1 : 0});
      if (!polled)
        return std::unexpected(polled.error());
      next_poll = now + std::chrono::milliseconds{1};
    }
    if (transport.interrupted())
      return false;
    const let accepted = io.receive(transport.received());
    const let consumed = transport.consume_received(accepted);
    if (!consumed)
      return std::unexpected(consumed.error());
    bool blocked = false;
    const let flushed = flush_consoles(services, io, &transport, &blocked);
    if (!flushed)
      return std::unexpected(flushed.error());
    if (!blocked && transport.pending_output() == 0)
      return true;
  }
}

fn dump_cpu(machine::instance &board, dynamips::devices::c7200_iofpga &io,
              std::optional<std::uint64_t> fault_pc = {}) -> void {
  const let &cpu = board.cpu();
  const let code_pc = fault_pc.value_or(cpu.pc.value);
  std::println(stderr, "CPU PC={:#x} EPC={:#x} code PC={:#x}", cpu.pc.value,
               cpu.cop0.registers[14], code_pc);
  let &uart = *io.descriptor().endpoint;
  const let status = uart.read(
      uart.context,
      {{0x1e84042c}, std::endian::big, jit::memory::access_size::byte});
  if (status)
    std::println(stderr, "UART ISR={:#x}", *status);
  std::println(stderr,
               "Stop Status={:#x} Cause={:#x} Count={:#x} Compare={:#x}",
               cpu.cop0.registers[12], cpu.cop0.registers[13],
               cpu.cop0.registers[9], cpu.cop0.registers[11]);
  for (unsigned i = 0; i < 32; i++)
    std::println(stderr, "r{}={:#x}", i, cpu.registers[i]);
  for (unsigned i = 0; i < 32; i++) {
    const let pc = code_pc - 64 + i * 4;
    const let word = board.memory().fetch<std::uint32_t>({pc});
    if (word)
      std::println(stderr, "{:#x}: {:#010x}", pc, *word);
  }
}

fn run(const char *rom_path, const char *ios_path,
         std::optional<std::uint64_t> budget, bool dump_ir,
         bool stop_at_handoff, std::optional<std::string_view> console_path,
         bool dump_state, bool stdio_console) -> int {
  let rom = load(rom_path);
  if (!rom)
    return 1;
  std::optional<dynamips::firmware::image> ios;
  if (ios_path) {
    let loaded = load(ios_path);
    if (!loaded)
      return 1;
    ios.emplace(std::move(*loaded));
  }
  dynamips::devices::rom_services services{
      {0x16000000},
      {.nvram = {0xffffffffbe000000ULL},
       .ios_entry = {ios ? ios->entry() : 0},
       .elf_machine_id = ios ? ios->machine_id() : std::uint16_t{0},
       .ram_megabytes = static_cast<std::uint32_t>(
           dynamips::c7200::npe400.ram_bytes / (1024 * 1024)),
       .configuration_register = dynamips::c7200::configuration_register,
       .rom_megabytes = static_cast<std::uint32_t>(
           dynamips::c7200::npe400.rom.size / (1024 * 1024)),
       .nvram_kibibytes = 128,
       .iomem_megabytes = 0}};
  std::array<std::byte, 128 * 1024> nvram_bytes{};
  dynamips::devices::byte_memory nvram{{0x1e000000}, nvram_bytes};
  let flash_storage = dynamips::host::allocate_aligned(
      jit::memory::page_size, dynamips::devices::bootflash::size);
  if (!flash_storage) {
    std::println(stderr, "boot: flash backing allocation failed: {}",
                 flash_storage.error().message());
    return 1;
  }
  std::fill_n(flash_storage->get(), dynamips::devices::bootflash::size,
              static_cast<unsigned char>(0xff));
  dynamips::devices::bootflash flash{
      {0x1a000000},
      std::as_writable_bytes(
          std::span{flash_storage->get(), dynamips::devices::bootflash::size})};
  dynamips::devices::c7200_environment environment;
  dynamips::devices::c7200_midplane midplane{&environment};
  dynamips::devices::c7200_iofpga io{dynamips::c7200::npe400_eeprom,
                                     dynamips::c7200::vxr_eeprom, &environment};
  dynamips::devices::clpd6729 pcmcia;
  dynamips::devices::gt64120 controller{{0x14000000}};
  if (!controller.configure_bridges(dynamips::c7200::npe400_bridges)) {
    std::println(stderr, "boot: invalid built-in PCI bridge configuration");
    return 1;
  }
  const std::array functions{pcmcia.configuration()};
  if (!controller.configure_functions(functions)) {
    std::println(stderr, "boot: invalid PCI function configuration");
    return 1;
  }
  const std::array attachments{
      machine::attachment{dynamips::c7200::gt64120, controller.descriptor()},
      machine::attachment{dynamips::c7200::rom_services, services.descriptor()},
      machine::attachment{dynamips::c7200::nvram, nvram.descriptor()},
      machine::attachment{dynamips::c7200::midplane_fpga,
                          midplane.descriptor()},
      machine::attachment{dynamips::c7200::io_fpga, io.descriptor(), 5},
      machine::attachment{dynamips::c7200::pci_io, pcmcia.descriptor()},
      machine::attachment{dynamips::c7200::bootflash, flash.descriptor()}};
  let board = machine::instance::create(
      dynamips::c7200::npe400, *rom, ios ? &*ios : nullptr, attachments,
      stop_at_handoff ? machine::boot_mode::handoff : machine::boot_mode::run);
  if (!board) {
    const let &error = board.error();
    if (const let *host = std::get_if<std::error_code>(&error.cause))
      std::println(stderr, "boot: machine setup failed: {}", host->message());
    else if (const let *issue = std::get_if<machine::setup_issue>(&error.cause))
      std::println(stderr, "boot: {} (item {})", setup_message(*issue),
                   error.item);
    else
      std::println(stderr, "boot: machine setup failed (category {}, item {})",
                   error.cause.index(), error.item);
    return 1;
  }
  std::unique_ptr<dynamips::host::console_transport> transport;
  if (console_path || stdio_console) {
    let listener =
        stdio_console
            ? dynamips::host::console_transport::terminal(true)
            : dynamips::host::console_transport::listen(*console_path, true);
    if (!listener) {
      std::println(stderr, "boot: cannot open console {}: {}",
                   console_path.value_or("stdio"), listener.error().message());
      return 1;
    }
    transport = std::move(*listener);
    if (console_path)
      std::println(stderr, "Console socket: {}", *console_path);
    else
      std::println(stderr, "Console: stdio; Ctrl-C goes to IOS, Ctrl-] exits.");
  }
  const let &model = (*board)->model();
  std::uint64_t executed = 0;
  std::chrono::steady_clock::time_point next_console_poll{};
  for (;;) {
    if (transport && !transport->connected()) {
      const let polled = transport->poll(std::chrono::milliseconds{10});
      if (!polled) {
        std::println(stderr, "boot: console I/O failed: {}",
                     polled.error().message());
        return 1;
      }
      if (transport->interrupted())
        return 0;
      if (!transport->connected())
        continue;
    }
    const let result = (*board)->run_slice(
        budget ? *budget - executed : std::numeric_limits<std::uint64_t>::max(),
        64, dump_ir);
    if (transport) {
      const let serviced =
          service_console(*transport, services, io, next_console_poll);
      if (!serviced) {
        std::println(stderr, "boot: console I/O failed: {}",
                     serviced.error().message());
        return 1;
      }
      if (!*serviced)
        return 0;
    } else if (const let flushed = flush_consoles(services, io); !flushed) {
      std::println(stderr, "boot: console output failed: {}",
                   flushed.error().message());
      return 1;
    }
    if (!result) {
      if (dump_state) {
        const let *guest =
            std::get_if<jit::dispatch_error>(&result.error().cause);
        dump_cpu(**board, io, guest ? std::optional{guest->pc} : std::nullopt);
      }
      if (const let *guest =
              std::get_if<jit::dispatch_error>(&result.error().cause)) {
        const let word = (*board)->memory().fetch<std::uint32_t>({guest->pc});
        if (guest->code == jit::dispatch_error_code::translation && word) {
          std::println(stderr,
                       "boot: unsupported instruction {:#010x} at PC {:#x}",
                       *word, guest->pc);
        } else if (guest->code == jit::dispatch_error_code::execution) {
          std::println(stderr, "boot: {} at PC {:#x}",
                       jit::amd64::message({guest->execution}), guest->pc);
        } else if (guest->code == jit::dispatch_error_code::memory) {
          std::println(stderr,
                       "boot blocked at PC {:#x}: memory fault {} at {:#x}",
                       guest->pc, static_cast<unsigned>(guest->access.reason),
                       guest->access.address.value);
          const let physical = machine::physical(guest->access.address.value);
          for (const let &window : model.windows)
            if (physical && *physical >= window.range.base.value &&
                *physical - window.range.base.value < window.range.size)
              std::println(stderr, "Device window: {}", window.name);
        } else if (guest->code == jit::dispatch_error_code::invalid_pc &&
                   !ios) {
          std::println(
              stderr,
              "boot: no IOS image loaded; pass --ios ELF to reach handoff");
        } else
          std::println(stderr, "boot blocked at PC {:#x}: dispatcher error {}",
                       guest->pc, static_cast<unsigned>(guest->code));
      } else {
        const let &device =
            std::get<jit::devices::service_error>(result.error().cause);
        std::println(stderr, "boot: device {} servicing failed: {}",
                     device.device.value, device.cause.message());
      }
      return 2;
    }
    if (budget)
      executed += result->instructions;
    if (dump_state && result->reason == jit::stop_reason::guest_exception)
      dump_cpu(**board, io, (*board)->cpu().cop0.registers[14]);
    if (result->reason == jit::stop_reason::yielded ||
        result->reason == jit::stop_reason::guest_exception)
      continue;
    if (result->reason == jit::stop_reason::handoff ||
        result->reason == jit::stop_reason::completed)
      return 0;
    if (dump_ir || dump_state)
      dump_cpu(**board, io);
    std::println(
        stderr,
        "boot: instruction budget exhausted at PC {:#x}; increase --budget",
        (*board)->cpu().pc.value);
    return 2;
  }
}
} // namespace

export namespace dynamips::boot {
struct options {
  const char *rom = "stable/mips64_microcode";
  const char *ios = nullptr;
  std::optional<std::uint64_t> budget;
  bool dump_ir = false;
  bool stop_at_handoff = false;
  std::optional<std::string_view> console_path;
  bool dump_state = false;
  bool stdio_console = false;
};

fn run(const options &config) -> int {
  try {
    if (config.stdio_console && config.console_path) {
      std::println(stderr,
                   "boot: stdio and socket consoles are mutually exclusive");
      return 1;
    }
    return ::run(config.rom, config.ios, config.budget, config.dump_ir,
                 config.stop_at_handoff, config.console_path, config.dump_state,
                 config.stdio_console);
  } catch (const std::exception &error) {
    std::println(stderr, "boot: {}", error.what());
    return 1;
  }
}
} // namespace dynamips::boot
