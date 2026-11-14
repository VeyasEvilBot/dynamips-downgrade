#include <array>
#include <bit>
#include <span>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <print>
#include <string_view>
#include <type_traits>

#include <dynamips/macros.h>

import jit.devices;
import jit.mips64;
import machine.interrupts;
import devices.c7200_iofpga;
import jit.memory;
import jit.uart;

namespace {
namespace cpu = jit::mips64;
namespace machine = dynamips::machine;
fn require(bool good, std::string_view message) -> void {
  if (!good) {
    std::println(stderr, "interrupt test: {}", message);
    std::exit(1);
  }
}

fn routing() -> void {
  cpu::cop0_state state{};
  constexpr std::uint64_t preserved = 0x80008320;
  state.registers[13] = preserved;
  state.registers[12] = 0x2001;
  {
    machine::interrupt_router router{state};
    let first = router.connect(5);
    let shared = router.connect(5);
    let separate = router.connect(3);
    require(first && shared && separate, "connect distinct and shared sources");
    require(state.registers[13] == preserved && !state.interrupt_pending(),
            "unasserted sources preserve software, timer and exception state");
    first->update(true);
    require(state.registers[13] == (preserved | 0x2000) &&
                state.interrupt_pending(),
            "external line asserts the selected CP0 pending bit immediately");
    shared->update(true);
    first->update(false);
    require((state.registers[13] & 0x2000) != 0,
            "clearing one source cannot lower a shared asserted line");
    separate->update(true);
    shared->update(false);
    require(state.registers[13] == (preserved | 0x800),
            "each input follows the OR of only its own sources");
    shared->update(false);
    require(state.registers[13] == (preserved | 0x800),
            "duplicate level notifications retain other lines");
    state.registers[12] = 0x8001;
    require(state.interrupt_pending(),
            "routing never clears the independently asserted timer interrupt");
    router.reset();
    require(state.registers[13] == preserved,
            "reset clears routed levels and preserves unowned Cause bits");
    separate->update(true);
    require(state.registers[13] == (preserved | 0x800),
            "source handles remain connected after a warm reset");
  }
  require(
      state.registers[13] == preserved,
      "router destruction clears its pending levels before releasing wiring");
}

fn limits() -> void {
  static_assert(!std::is_copy_constructible_v<machine::interrupt_router>);
  static_assert(!std::is_move_constructible_v<machine::interrupt_router>);
  cpu::cop0_state state{};
  machine::interrupt_router router{state};
  for (const std::uint8_t line : std::array<std::uint8_t, 5>{0, 1, 7, 8, 255}) {
    const let invalid = router.connect(line);
    require(!invalid &&
                invalid.error() == machine::interrupt_error::invalid_line,
            "software inputs, timer input and out-of-range lines are rejected");
  }
  std::array<jit::devices::interrupt_line, machine::interrupt_router::capacity>
      sources{};
  for (std::size_t i = 0; i < sources.size(); i++) {
    let connected = router.connect(static_cast<std::uint8_t>(2 + i % 5));
    require(connected.has_value(), "connect every bounded source slot");
    sources[i] = *connected;
  }
  const let full = router.connect(5);
  require(!full && full.error() == machine::interrupt_error::capacity_exhausted,
          "capacity exhaustion is a typed failure");
  for (const let &source : sources)
    source.update(true);
  require(state.registers[13] == 0x7c00,
          "allocated source contexts stay stable while later sources connect");
  router.reset();
  require(state.registers[13] == 0,
          "reset clears all routed pending bits without invalidating handles");
  sources.front().update(true);
  require(state.registers[13] == 0x400,
          "first source remains stable after allocation and reset");
  const let still_full = router.connect(2);
  require(!still_full && still_full.error() ==
                             machine::interrupt_error::capacity_exhausted,
          "warm reset retains wiring and does not recycle borrowed handles");
}
fn uart_levels() -> void {
  cpu::cop0_state state{};
  state.registers[13] = 0x80008024;
  machine::interrupt_router router{state};
  const let line = router.connect(5);
  require(line.has_value(), "route console to CP0 input five");
  dynamips::devices::c7200_iofpga io;
  const let device = io.descriptor();
  device.lifecycle.connect_interrupt(device.lifecycle.context, *line);
  let &endpoint = *device.endpoint;
  const let access = [](unsigned offset) {
    return jit::memory::device_access{{0x1e840000 + offset},
                                      std::endian::big,
                                      jit::memory::access_size::word};
  };
  const let write = [&](unsigned offset, unsigned value) {
    require(endpoint.write(endpoint.context, access(offset), value).has_value(),
            "write console control");
  };
  write(0x414, 5);
  write(0x42c, 1);
  require((state.registers[13] & 0x2000) &&
              endpoint.read(endpoint.context, access(0x42c)) == 1,
          "enabled empty transmitter asserts a level interrupt");
  state.write(13, 0x100);
  require(state.registers[13] == 0x8000a124,
          "MTC0 Cause changes only software pending bits");
  write(0x42c, 2);
  require(!(state.registers[13] & 0x2000),
          "masking TX deasserts its interrupt");
  const std::array input_bytes{std::byte{'x'}, std::byte{'y'}};
  require(io.receive(input_bytes) == 2 && (state.registers[13] & 0x2000),
          "console RX asserts the masked input source");
  require(endpoint.read(endpoint.context, access(0x41c)) == 'x' &&
              (state.registers[13] & 0x2000) &&
              endpoint.read(endpoint.context, access(0x41c)) == 'y' &&
              !(state.registers[13] & 0x2000),
          "RX remains asserted until the last queued byte is consumed");
  require(io.receive(input_bytes) == 2, "queue data before receiver disable");
  write(0x414, 2);
  require(io.receive(input_bytes) == 0 &&
              (endpoint.read(endpoint.context, access(0x40c)).value() & 1) &&
              (state.registers[13] & 0x2000),
          "receiver disable retains unread FIFO status and interrupt");
  require(endpoint.read(endpoint.context, access(0x41c)) == 'x' &&
              endpoint.read(endpoint.context, access(0x41c)) == 'y' &&
              !(state.registers[13] & 0x2000),
          "disabled receiver still permits reading its pending bytes");
  write(0x414, 5);
  write(0x42c, 1);
  for (unsigned i = 0; i < jit::uart::capacity; i++)
    write(0x41c, 'x');
  require(!(state.registers[13] & 0x2000), "full TX queue removes readiness");
  const let drained =
      io.drain(false,
               [](std::span<const std::byte> bytes) noexcept
                   -> std::expected<std::size_t, int> { return bytes.size(); });
  require(drained && (state.registers[13] & 0x2000),
          "draining TX restores readiness and its interrupt");
  write(0x41c, 'z');
  write(0x414, 8);
  require(!(state.registers[13] & 0x2000),
          "TX disable removes its ready interrupt");
  const let final_byte =
      io.drain(false,
               [](std::span<const std::byte> data) noexcept
                   -> std::expected<std::size_t, int> { return data.size(); });
  require(final_byte && *final_byte == 1,
          "accepted output completes after transmitter disable");
  write(0x41c, 'w');
  write(0x414, 4);
  const let ignored =
      io.drain(false,
               [](std::span<const std::byte> data) noexcept
                   -> std::expected<std::size_t, int> { return data.size(); });
  require(ignored && *ignored == 0,
          "disabled transmitter writes do not queue a later character");
  device.lifecycle.connect_interrupt(device.lifecycle.context, {});
  require(!(state.registers[13] & 0x2000),
          "disconnect clears the borrowed line");
  io.reset();
  require(!(state.registers[13] & 0x2000),
          "disconnected reset cannot signal CPU");
}
} // namespace

fn main() -> int {
  routing();
  uart_levels();
  limits();
  std::println("interrupts: wired OR, CP0 preservation, reset and bounded "
               "sources passed");
}
