#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <new>
#include <span>
#include <charconv>
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

import devices.byte_memory;
import devices.rom_services;
import machine.firmware;
import machine.core;
import machine.c7200;
import jit.memory;
import jit.dispatch;
import jit.devices;

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

fn flush_output(dynamips::devices::rom_services &device)
    -> std::expected<void, std::error_code> {
  using dynamips::devices::rom_output;
  for (const let channel : {rom_output::console, rom_output::log}) {
    FILE *stream = channel == rom_output::console ? stdout : stderr;
    const let drained = device.drain(
        channel,
        [stream, channel](std::span<const std::byte> bytes) noexcept
            -> std::expected<std::size_t, std::error_code> {
          if (channel == rom_output::log)
            return bytes.size();
          try {
            std::print(
                stream, "{}",
                std::string_view{reinterpret_cast<const char *>(bytes.data()),
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
    if (channel == rom_output::console && *drained != 0 &&
        std::fflush(stream) != 0)
      return std::unexpected(std::error_code{errno, std::generic_category()});
  }
  return {};
}

fn run(const char *rom_path, const char *ios_path, std::uint64_t budget,
         bool dump_ir) -> int {
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
      {0x16000000}, {{0xffffffffbe000000ULL}, {ios ? ios->entry() : 0}}};
  std::array<std::byte, 128 * 1024> nvram_bytes{};
  dynamips::devices::byte_memory nvram{{0x1e000000}, nvram_bytes};
  const std::array attachments{
      machine::attachment{dynamips::c7200::rom_services, services.descriptor()},
      machine::attachment{dynamips::c7200::nvram, nvram.descriptor()}};
  let board = machine::instance::create(dynamips::c7200::npe400, *rom,
                                         ios ? &*ios : nullptr, attachments);
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
  const let &model = (*board)->model();
  std::uint64_t executed = 0;
  for (;;) {
    const let result = (*board)->run_slice(budget - executed, 64, dump_ir);
    if (const let flushed = flush_output(services); !flushed) {
      std::println(stderr, "boot: console output failed: {}",
                   flushed.error().message());
      return 1;
    }
    if (!result) {
      if (const let *guest =
              std::get_if<jit::dispatch_error>(&result.error().cause)) {
        const let word = (*board)->memory().fetch<std::uint32_t>({guest->pc});
        if (guest->code == jit::dispatch_error_code::translation && word) {
          std::println(
              stderr,
              "boot blocked at PC {:#x}: {} ({:#010x}), translation error {}",
              guest->pc, "unsupported opcode", *word,
              static_cast<unsigned>(guest->translation.code));
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
    executed += result->instructions;
    if (result->reason == jit::stop_reason::yielded)
      continue;
    if (result->reason == jit::stop_reason::handoff ||
        result->reason == jit::stop_reason::completed)
      return 0;
    std::println(
        stderr,
        "boot: instruction budget exhausted at PC {:#x}; increase --budget",
        (*board)->cpu().pc.value);
    return 2;
  }
}
} // namespace

fn main(int argc, char **argv) -> int {
  try {
    const char *rom = "stable/mips64_microcode";
    const char *ios = nullptr;
    std::uint64_t budget = 4096;
    bool dump_ir = false;
    for (int i = 1; i < argc; i++) {
      const std::string_view argument{argv[i]};
      if (argument == "--help") {
        std::println(
            "Usage: c7200-boot [--rom ELF] [--ios ELF_OR_ZIP] [--budget N] "
            "[--dump-ir]\n"
            "Prepare the C++ NPE-400 chassis and probe ROM execution.\n"
            "Default ROM: stable/mips64_microcode. Exit 2 means boot is "
            "blocked; exit 0 includes a successful IOS handoff.");
        return 0;
      }
      if (argument == "--dump-ir") {
        dump_ir = true;
        continue;
      }
      if ((argument != "--rom" && argument != "--ios" &&
           argument != "--budget") ||
          i + 1 == argc) {
        std::println(stderr, "boot: invalid option or missing value: {}",
                     argument);
        return 1;
      }
      i++;
      if (argument == "--rom")
        rom = argv[i];
      else if (argument == "--ios")
        ios = argv[i];
      else {
        const std::string_view text{argv[i]};
        const let parsed =
            std::from_chars(text.data(), text.data() + text.size(), budget);
        if (text.empty() || parsed.ec != std::errc{} ||
            parsed.ptr != text.data() + text.size()) {
          std::println(stderr, "boot: invalid instruction budget: {}", text);
          return 1;
        }
      }
    }
    return run(rom, ios, budget, dump_ir);
  } catch (const std::exception &error) {
    std::println(stderr, "boot: {}", error.what());
    return 1;
  }
}
