#include <bit>
#include <charconv>
#include <chrono>
#include <memory>
#include <new>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <cstdio>
#include <expected>
#include <system_error>
#include <print>
#include <span>
#include <string_view>
#include <vector>
#include <optional>
#include <variant>
#include <utility>

#include <dynamips/macros.h>

import jit.mips64;
import jit.amd64;
import jit.memory;
import jit.uart;
import jit.example;
import jit.dispatch;
import jit.devices;
import jit.execution;
import host_memory;
import host.console;
import devices.console;

namespace {
namespace mips = jit::mips64;

fn fault_name(jit::memory::fault_reason reason) noexcept -> std::string_view {
  switch (reason) {
  case jit::memory::fault_reason::misaligned:
    return "misaligned access";
  case jit::memory::fault_reason::unmapped:
    return "unmapped address";
  case jit::memory::fault_reason::permission:
    return "permission denied";
  case jit::memory::fault_reason::unsupported_size:
    return "unsupported access width";
  case jit::memory::fault_reason::bus:
    return "device bus fault";
  }
  return "unknown memory fault";
}

fn dispatch_message(const jit::dispatch_error &error) noexcept
    -> std::string_view {
  switch (error.code) {
  case jit::dispatch_error_code::invalid_image:
    return "invalid instruction image";
  case jit::dispatch_error_code::invalid_pc:
    return "PC outside instruction image";
  case jit::dispatch_error_code::code_changed:
    return "code changed; recreate dispatcher";
  case jit::dispatch_error_code::writable_code:
    return "executable RAM has a writable alias";
  case jit::dispatch_error_code::compilation:
    return jit::amd64::message(error.compilation);
  case jit::dispatch_error_code::memory:
    return fault_name(error.access.reason);
  case jit::dispatch_error_code::out_of_memory:
    return "dispatcher allocation failed";
  case jit::dispatch_error_code::diagnostic:
    return "cannot print IR";
  case jit::dispatch_error_code::translation:
    switch (error.translation.code) {
    case mips::error_code::unaligned_pc:
      return "unaligned PC";
    case mips::error_code::block_too_large:
      return "block too large";
    case mips::error_code::unsupported_instruction:
      return "unsupported instruction";
    case mips::error_code::invalid_encoding:
      return "invalid instruction encoding";
    case mips::error_code::out_of_memory:
      return "IR scratch area exhausted";
    case mips::error_code::invalid_ir:
      return "invalid IR";
    case mips::error_code::missing_delay_slot:
      return "branch has no delay slot";
    case mips::error_code::branch_in_delay_slot:
      return "branch in delay slot is unsupported";
    }
  }
  return "dispatch failed";
}

struct uart_io {
  jit::uart &console;
  std::span<const std::byte> input;

  static fn reset(void *context) noexcept -> void {
    static_cast<uart_io *>(context)->console.reset();
  }

  static fn service(void *context) noexcept
      -> std::expected<void, std::error_code> {
    let &self = *static_cast<uart_io *>(context);
    self.input = self.input.subspan(self.console.receive(self.input));
    const let drained =
        self.console.drain([](std::span<const std::byte> bytes) noexcept
                               -> std::expected<std::size_t, std::error_code> {
          try {
            std::print(
                stdout, "{}",
                std::string_view{reinterpret_cast<const char *>(bytes.data()),
                                 bytes.size()});
            return bytes.size();
          } catch (const std::bad_alloc &) {
            return std::unexpected(
                std::make_error_code(std::errc::not_enough_memory));
          } catch (const std::system_error &error) {
            return std::unexpected(error.code());
          }
        });
    if (!drained || (*drained != 0 && std::fflush(stdout) != 0))
      return std::unexpected(std::make_error_code(std::errc::io_error));
    return {};
  }
};

fn run(std::span<const mips::instruction_word> words, bool dump_ir,
         bool registers, bool example = false, std::string_view input = {},
         std::optional<std::string_view> console_path = std::nullopt) -> int {
  constexpr mips::guest_address start{0x1000};
  if (words.size() > jit::memory::page_size / sizeof(std::uint32_t)) {
    std::println(stderr,
                 "jit: program exceeds the 1024-instruction playground limit");
    return 2;
  }
  constexpr std::size_t ram_size = 2 * jit::memory::page_size;
  let ram = dynamips::host::allocate_aligned(jit::memory::page_size, ram_size);
  if (!ram) {
    std::println(stderr, "jit: cannot allocate guest RAM: {}",
                 ram.error().message());
    return 1;
  }
  std::memset(ram->get(), 0, ram_size);
  std::unique_ptr<dynamips::host::console_transport> transport;
  if (console_path) {
    let listener =
        dynamips::host::console_transport::listen(*console_path, true);
    if (!listener) {
      std::println(stderr, "jit: cannot listen on {}: {}", *console_path,
                   listener.error().message());
      return 1;
    }
    transport = std::move(*listener);
    std::println("Console socket: {}", *console_path);
    if (std::fflush(stdout) != 0)
      return 1;
  }
  jit::uart console{{0x100000}};
  uart_io io{console, std::as_bytes(std::span{input.data(), input.size()})};
  std::optional<jit::devices::uart_console> live_console;
  if (transport)
    live_console.emplace(console, *transport);
  jit::devices::bus devices;
  if (!devices.add(
          {{0x100000}, 12},
          {&console.endpoint(), jit::devices::widths::word, std::endian::big,
           live_console
               ? live_console->lifecycle()
               : jit::devices::hooks{&io, uart_io::reset, uart_io::service}})) {
    std::println(stderr, "jit: cannot register UART");
    return 1;
  }
  devices.reset();
  let address_space = jit::memory::address_space::create(
      std::as_writable_bytes(std::span{ram->get(), ram_size}), 3);
  if (!address_space) {
    std::println(stderr, "jit: cannot reserve guest memory");
    return 1;
  }
  let &guest = **address_space;
  constexpr let rw =
      jit::memory::permission::read | jit::memory::permission::write;
  if (!guest.map_ram({start.value}, {jit::memory::page_size}, rw) ||
      !guest.map_ram({0x2000}, {0}, rw) ||
      !guest.map_device({0x8000}, {0x100000}, devices.endpoint(), rw)) {
    std::println(stderr, "jit: cannot map playground RAM and MMIO");
    return 1;
  }
  for (std::size_t i = 0; i < words.size(); i++)
    if (!guest.store<std::uint32_t>({start.value + i * 4}, words[i].bits)) {
      std::println(stderr, "jit: cannot load program into guest RAM");
      return 1;
    }
  if (!guest.protect({start.value}, jit::memory::permission::read |
                                        jit::memory::permission::execute)) {
    std::println(stderr, "jit: cannot protect guest code");
    return 1;
  }
  let dispatch = jit::dispatcher::create(words, start, guest);
  if (!dispatch) {
    std::println(stderr, "jit: {}", dispatch_message(dispatch.error()));
    return 1;
  }
  jit::execution_loop loop{**dispatch, devices};
  mips::state state{};
  state.pc = start;
  constexpr std::uint64_t budget = 1'000'000;
  std::uint64_t instructions = 0;
  if (!transport)
    std::println("MMIO console:");
  while (true) {
    if (transport) {
      const let polled = transport->poll(
          std::chrono::milliseconds{transport->connected() ? 1 : 10});
      if (!polled) {
        std::println(stderr, "jit: console I/O failed: {}",
                     polled.error().message());
        return 1;
      }
      if (transport->interrupted())
        break;
      if (!transport->connected())
        continue;
      if (transport->input_closed() && transport->received().empty() &&
          console.pending_input() == 0 && console.pending() == 0 &&
          transport->pending_output() == 0)
        break;
    }
    const let executed = loop.run_slice(
        state, transport ? 4096 : budget - instructions, 64, dump_ir);
    if (!executed) {
      const let &failure = executed.error();
      if (failure.service_after_fault)
        std::println(stderr, "jit: device {} servicing also failed: {}",
                     failure.service_after_fault->device.value,
                     failure.service_after_fault->cause.message());
      const let *guest_error = std::get_if<jit::dispatch_error>(&failure.cause);
      if (!guest_error) {
        const let &error = std::get<jit::devices::service_error>(failure.cause);
        std::println(stderr, "jit: device {} servicing failed: {}",
                     error.device.value, error.cause.message());
        return 1;
      }
      const let &error = *guest_error;
      if (error.code == jit::dispatch_error_code::memory)
        std::println(stderr, "jit: {} at PC {:#x}, address {:#x}",
                     fault_name(error.access.reason), error.pc,
                     error.access.address.value);
      else
        std::println(stderr, "jit: {} at PC {:#x}", dispatch_message(error),
                     error.pc);
      if (error.branch_pc)
        std::println(stderr, "jit: delay slot of branch at {:#x}",
                     *error.branch_pc);
      if (registers)
        state.print();
      return 1;
    }
    instructions += executed->instructions;
    if (executed->reason == jit::stop_reason::completed)
      break;
    if (!transport && executed->reason == jit::stop_reason::budget_exhausted) {
      std::println(stderr, "jit: instruction budget exhausted at PC {:#x}",
                   state.pc.value);
      return 2;
    }
  }
  if (transport) {
    const let closed = transport->close();
    if (!closed) {
      std::println(stderr, "jit: console close failed: {}",
                   closed.error().message());
      return 1;
    }
  }
  std::println("\nExecuted {} instructions; PC={:#x}", instructions,
               state.pc.value);
  if (example) {
    const let sum = guest.load<std::uint64_t>({0x2010});
    const let check = guest.load<std::uint32_t>({0x2018});
    const let ready = guest.load<std::uint32_t>({0x201c});
    if (!sum || !check || !ready) {
      std::println(stderr, "jit: cannot inspect example results");
      return 1;
    }
    std::println("RAM: sum={}, readback XOR={}, UART ready={}", *sum, *check,
                 *ready);
  }
  if (registers)
    state.print();
  return 0;
}
} // namespace

fn main(int argc, char **argv) -> int {
  try {
    bool dump_ir = false;
    bool registers = false;
    std::optional<std::string_view> echo;
    std::optional<std::string_view> console_path;
    std::vector<mips::instruction_word> words;
    words.reserve(static_cast<std::size_t>(argc - 1));
    for (int i = 1; i < argc; i++) {
      std::string_view text{argv[i]};
      if (text == "--help") {
        std::println(
            "Usage: jit-playground [--dump-ir] [--registers] [--console PATH | "
            "--echo TEXT | "
            "HEX_WORD ...]\n"
            "Run the RAM sum/MMIO example, or a bounded MIPS program.\n"
            "Code: 0x1000; RAM: 0x2000; MMIO TX: 0x8000; status: 0x8004.\n"
            "Supported: NOP, BEQ, BNE, ORI, XOR, DADDU, LW, SW, LD, SD. "
            "Registers start at "
            "zero.");
        return 0;
      }
      if (text == "--console") {
        if (console_path || i + 1 == argc) {
          std::println(stderr, "jit: --console requires one socket path");
          return 2;
        }
        i++;
        console_path = argv[i];
        continue;
      }
      if (text == "--echo") {
        if (echo || i + 1 == argc) {
          std::println(stderr, "jit: --echo requires one text argument");
          return 2;
        }
        i++;
        echo = argv[i];
        continue;
      }
      if (text == "--dump-ir") {
        dump_ir = true;
        continue;
      }
      if (text == "--registers") {
        registers = true;
        continue;
      }
      if (text.starts_with("0x") || text.starts_with("0X"))
        text.remove_prefix(2);
      std::uint32_t bits = 0;
      const let parsed =
          std::from_chars(text.data(), text.data() + text.size(), bits, 16);
      if (text.empty() || parsed.ec != std::errc{} ||
          parsed.ptr != text.data() + text.size()) {
        std::println(stderr, "jit: invalid 32-bit hexadecimal word: {}",
                     argv[i]);
        return 2;
      }
      words.push_back({bits});
    }
    if (console_path) {
      if (echo || !words.empty()) {
        std::println(stderr, "jit: --console runs live echo and cannot be "
                             "combined with text or instruction words");
        return 2;
      }
      const let program = jit::example::echo_loop();
      return run(program, dump_ir, registers, false, {}, console_path);
    }
    if (echo) {
      if (!words.empty() || echo->size() > 65535) {
        std::println(
            stderr,
            "jit: --echo accepts at most 65535 bytes and no instruction words");
        return 2;
      }
      const let program =
          jit::example::echo_program(static_cast<std::uint16_t>(echo->size()));
      return run(program, dump_ir, registers, false, *echo);
    }
    if (!words.empty())
      return run(words, dump_ir, registers);
    return run(jit::example::program, dump_ir, registers, true);
  } catch (const std::exception &error) {
    std::println(stderr, "jit: {}", error.what());
    return 1;
  }
}
