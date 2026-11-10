#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <print>
#include <span>
#include <string_view>
#include <system_error>
#include <variant>

#include <dynamips/macros.h>

import jit.memory;
import jit.uart;
import jit.devices;
import jit.dispatch;
import jit.execution;
import jit.example;
import jit.mips64;

namespace {
namespace mem = jit::memory;
namespace dev = jit::devices;
constexpr let rw = mem::permission::read | mem::permission::write;
constexpr let rx = mem::permission::read | mem::permission::execute;

fn require(bool condition, std::string_view message) -> void {
  if (!condition) {
    std::println(stderr, "device test: {}", message);
    std::exit(1);
  }
}

fn routing() -> void {
  jit::uart first{{0x100000}};
  jit::uart second{{0x100010}};
  dev::bus bus;
  const dev::descriptor a{&first.endpoint(),
                          dev::widths::word,
                          std::endian::big,
                          {&first,
                           [](void *context) noexcept {
                             static_cast<jit::uart *>(context)->reset();
                           },
                           nullptr}};
  const dev::descriptor b{&second.endpoint(), dev::widths::word,
                          std::endian::big};
  require(bus.add({{0x100000}, 12}, a).has_value() &&
              bus.add({{0x100010}, 12}, b).has_value(),
          "register adjacent devices");
  require(bus.add({{0x100008}, 12}, b) ==
              std::unexpected(dev::registration_error::overlap),
          "reject overlapping register ranges");
  require(!bus.add({{UINT64_MAX}, 2}, b) && !bus.add({{0}, 0}, b),
          "reject overflowing and empty register ranges");
  const dev::descriptor invalid{nullptr, dev::widths::word, std::endian::big};
  require(!bus.add({{0}, 4}, invalid), "reject missing endpoint");
  unsigned accesses = 0;
  mem::device registers{&accesses,
                        [](void *context, mem::device_access access) noexcept
                            -> std::expected<std::uint64_t, mem::device_error> {
                          require(access.order == std::endian::little,
                                  "callback receives declared register order");
                          (*static_cast<unsigned *>(context))++;
                          return 0x12345678;
                        },
                        nullptr};
  require(bus.add({{0x100020}, 6},
                  {&registers, dev::widths::word, std::endian::little})
              .has_value(),
          "register little endian read-only registers");
  let memory = mem::address_space::create({}, 1);
  require(memory && (*memory)
                        ->map_device({0x8000}, {0x100000}, bus.endpoint(), rw)
                        .has_value(),
          "map shared device page");
  require(bus.add({{0x200000}, 12}, a) ==
              std::unexpected(dev::registration_error::sealed),
          "routing table freezes before exposure to memory");
  require(
      (*memory)->store<std::uint32_t>({0x8000}, 'A').has_value() &&
          (*memory)
              ->store<std::uint32_t, std::endian::little>({0x8010}, 0x42000000)
              .has_value(),
      "route big and little endian writes");
  require(first.output() == "A" && second.output() == "B",
          "devices remain isolated");
  require((*memory)->load<std::uint32_t, std::endian::little>({0x8004}) ==
              0x01000000,
          "bus converts register values to guest byte order");
  const let gap = (*memory)->load<std::uint32_t>({0x800c});
  const let width = (*memory)->load<std::uint16_t>({0x8004});
  require(!gap && gap.error().reason == mem::fault_reason::bus && !width &&
              width.error().reason == mem::fault_reason::unsupported_size,
          "reject register gaps and unsupported widths");
  require((*memory)->load<std::uint32_t>({0x8020}) == 0x78563412 &&
              (*memory)->load<std::uint32_t, std::endian::little>({0x8020}) ==
                  0x12345678,
          "little endian register device supports either guest access order");
  const let crossing = (*memory)->load<std::uint32_t>({0x8024});
  const let readonly = (*memory)->store<std::uint32_t>({0x8020}, 0);
  require(!crossing && crossing.error().reason == mem::fault_reason::bus &&
              !readonly && readonly.error().reason == mem::fault_reason::bus &&
              accesses == 2,
          "reject crossing a device boundary and writes to read-only registers "
          "before callback");
  constexpr std::array input{std::byte{0x61}};
  require(first.receive(input) == 1, "queue input before reset");
  bus.reset();
  require(first.pending() == 0 &&
              (*memory)->load<std::uint32_t>({0x8004}) == 1 &&
              second.output() == "B",
          "reset clears only the device with a reset hook");
  dev::bus full;
  for (std::size_t i = 0; i < dev::bus::capacity; i++)
    require(full.add({{i * 16}, 12}, b).has_value(), "fill bounded bus");
  require(full.add({{0x1000}, 12}, b) ==
              std::unexpected(dev::registration_error::full),
          "bounded registration does not allocate overflow storage");
}

struct echo_io {
  jit::uart console{{0x100000}};
  std::array<std::byte, 5000> input{};
  std::array<std::byte, 5000> output{};
  std::size_t sent = 0;
  std::size_t received = 0;
  unsigned calls = 0;
  unsigned fail_at = 0;

  static fn reset(void *context) noexcept -> void {
    let &self = *static_cast<echo_io *>(context);
    self.console.reset();
    self.sent = self.received = self.calls = 0;
  }

  static fn service(void *context) noexcept
      -> std::expected<void, std::error_code> {
    let &self = *static_cast<echo_io *>(context);
    self.calls++;
    if (self.calls == self.fail_at)
      return std::unexpected(std::make_error_code(std::errc::io_error));
    const let input = std::span{self.input}.subspan(self.sent);
    self.sent += self.console.receive(
        input.first(std::min<std::size_t>(17, input.size())));
    bool acknowledged = false;
    const let drained =
        self.console.drain([&](std::span<const std::byte> bytes) noexcept
                               -> std::expected<std::size_t, std::error_code> {
          if (acknowledged)
            return 0;
          acknowledged = true;
          const let count = std::min<std::size_t>(5, bytes.size());
          require(count <= self.output.size() - self.received,
                  "bounded host capture");
          std::copy_n(bytes.begin(), count,
                      self.output.begin() +
                          static_cast<std::ptrdiff_t>(self.received));
          self.received += count;
          return count;
        });
    if (!drained)
      return std::unexpected(std::make_error_code(std::errc::io_error));
    return {};
  }
};

fn attach(dev::bus &bus, echo_io &io, mem::address_space &guest) -> void {
  require(bus.add({{0x100000}, 12}, {&io.console.endpoint(),
                                     dev::widths::word,
                                     std::endian::big,
                                     {&io, echo_io::reset, echo_io::service}})
                  .has_value() &&
              guest.map_device({0x8000}, {0x100000}, bus.endpoint(), rw)
                  .has_value(),
          "attach UART bus and servicing hooks");
  bus.reset();
}

fn load(mem::address_space &guest,
          std::span<const jit::mips64::instruction_word> words) -> void {
  require(guest.map_ram({0x1000}, {0}, rw).has_value(), "map guest code");
  for (std::size_t i = 0; i < words.size(); i++)
    require(
        guest.store<std::uint32_t>({0x1000 + i * 4}, words[i].bits).has_value(),
        "load guest code");
  require(guest.protect({0x1000}, rx).has_value(), "protect guest code");
}

fn echo() -> void {
  std::array<std::byte, mem::page_size> ram{};
  echo_io io;
  for (std::size_t i = 0; i < io.input.size(); i++)
    io.input[i] = static_cast<std::byte>(i & 255);
  dev::bus bus;
  let memory = mem::address_space::create(ram, 2);
  require(memory.has_value(), "create echo memory");
  attach(bus, io, **memory);
  const let words = jit::example::echo_program(5000);
  load(**memory, words);
  let dispatch = jit::dispatcher::create(words, {0x1000}, **memory);
  require(dispatch.has_value(), "create echo dispatcher");
  jit::execution_loop loop{**dispatch, bus};
  jit::mips64::state state{};
  state.pc = {0x1000};
  std::uint64_t instructions = 0;
  bool completed = false;
  for (unsigned i = 0; i < 2000; i++) {
    const let result = loop.run_slice(state, 1'000'000 - instructions, 16);
    require(result.has_value(), "service and execute real echo guest");
    instructions += result->instructions;
    completed = result->reason == jit::stop_reason::completed;
    if (completed && io.console.pending() == 0)
      break;
    require(result->reason != jit::stop_reason::budget_exhausted,
            "echo makes progress");
  }
  require(completed && io.sent == io.input.size() &&
              io.received == io.input.size() && io.output == io.input &&
              state.registers[5] == 5000,
          "guest echoes streamed RX through TX across FIFO capacity and short "
          "drains");
}

fn service_failures() -> void {
  std::array<std::byte, mem::page_size> ram{};
  echo_io io;
  dev::bus bus;
  let memory = mem::address_space::create(ram, 2);
  require(memory.has_value(), "create fault memory");
  attach(bus, io, **memory);
  constexpr std::array words{jit::mips64::instruction_word{0x34018000},
                             jit::mips64::instruction_word{0x34020058},
                             jit::mips64::instruction_word{0xac220000},
                             jit::mips64::instruction_word{0x34039000},
                             jit::mips64::instruction_word{0x8c620000}};
  load(**memory, words);
  let dispatch = jit::dispatcher::create(words, {0x1000}, **memory);
  require(dispatch.has_value(), "create fault dispatcher");
  jit::execution_loop loop{**dispatch, bus};
  jit::mips64::state state{};
  state.pc = {0x1000};
  io.fail_at = 1;
  const let before = loop.run_slice(state, 100);
  require(
      !before &&
          std::holds_alternative<dev::service_error>(before.error().cause) &&
          state.pc.value == 0x1000 && state.registers[1] == 0,
      "pre-service failure prevents guest execution");
  io.fail_at = 3;
  const let both = loop.run_slice(state, 100);
  require(!both &&
              std::holds_alternative<jit::dispatch_error>(both.error().cause) &&
              both.error().service_after_fault && state.pc.value == 0x1010 &&
              io.console.output() == "X",
          "guest fault and failed output drain are both retained");
  require(bus.service().has_value() && io.received == 1 &&
              io.output[0] == std::byte{'X'},
          "failed service preserves committed guest output for retry");
  io.console.reset();
  io.received = 0;
  state = {};
  state.pc = {0x1000};
  const let guest_fault = loop.run_slice(state, 100);
  require(!guest_fault && !guest_fault.error().service_after_fault &&
              io.received == 1 && io.console.pending() == 0,
          "post-service drains output even when the guest faults");
}
} // namespace

fn main() -> int {
  routing();
  echo();
  service_failures();
  std::println(
      "devices: MMIO bus, reset, queued echo and service failures passed");
}
