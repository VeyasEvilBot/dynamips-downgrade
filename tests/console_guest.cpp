#include <array>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <print>
#include <span>
#include <utility>
#include <vector>

#include <dynamips/macros.h>

import host.console;
import native_fd;
import jit.memory;
import jit.uart;
import jit.devices;
import jit.dispatch;
import jit.execution;
import jit.example;
import jit.mips64;
import devices.console;

namespace {
fn check(bool condition, const char *message) -> bool {
  if (!condition)
    std::println(stderr, "console guest test: {}", message);
  return condition;
}
} // namespace

fn console_guest_echo(
    std::unique_ptr<dynamips::host::console_transport> server,
    dynamips::host::file_descriptor descriptor) -> bool {
  namespace mem = jit::memory;
  let client = dynamips::host::console_transport::adopt(std::move(descriptor));
  if (!check(client.has_value(), "adopt real client stream"))
    return false;
  std::array<std::byte, mem::page_size> ram{};
  jit::uart uart{{0x100000}};
  jit::devices::uart_console adapter{uart, *server};
  jit::devices::bus bus;
  let memory = mem::address_space::create(ram, 2);
  if (!check(memory.has_value(), "create live guest memory"))
    return false;
  let &guest = **memory;
  constexpr let rw = mem::permission::read | mem::permission::write;
  if (!check(bus.add({{0x100000}, 12},
                     {&uart.endpoint(), jit::devices::widths::word,
                      std::endian::big, adapter.lifecycle()})
                     .has_value() &&
                 guest.map_device({0x8000}, {0x100000}, bus.endpoint(), rw)
                     .has_value() &&
                 guest.map_ram({0x1000}, {0}, rw).has_value(),
             "map production UART adapter"))
    return false;
  bus.reset();
  const let words = jit::example::echo_loop();
  for (std::size_t i = 0; i < words.size(); i++)
    if (!check(guest.store<std::uint32_t>({0x1000 + i * 4}, words[i].bits)
                   .has_value(),
               "load live echo guest"))
      return false;
  if (!check(guest
                 .protect({0x1000},
                          mem::permission::read | mem::permission::execute)
                 .has_value(),
             "protect live echo guest"))
    return false;
  let dispatch = jit::dispatcher::create(words, {0x1000}, guest);
  if (!check(dispatch.has_value(), "compile live guest dispatch"))
    return false;
  jit::execution_loop loop{**dispatch, bus};
  jit::mips64::state state{};
  state.pc = {0x1000};
  std::vector<std::byte> input(32768), output(input.size());
  for (std::size_t i = 0; i < input.size(); i++)
    input[i] = static_cast<std::byte>((i * 11) & 255);
  std::size_t sent = 0, received = 0;
  bool rx_blocked = false, tx_blocked = false;
  const let deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds{10};
  while (std::chrono::steady_clock::now() < deadline) {
    sent += (*client)->send(std::span{input}.subspan(sent));
    if (!check((*client)->poll().has_value(), "send real guest input"))
      return false;
    const let result = loop.run_slice(state, 4096, 64);
    if (!check(result.has_value(),
               "execute guest through real console service hooks"))
      return false;
    rx_blocked |= uart.pending_input() == jit::uart::capacity &&
                  !server->received().empty();
    tx_blocked |= uart.pending() == jit::uart::capacity;
    if (!check((*client)->poll().has_value(), "poll real guest output"))
      return false;
    const let bytes = (*client)->received();
    const let count = std::min<std::size_t>(17, bytes.size());
    if (!check(count <= output.size() - received, "bound echoed bytes"))
      return false;
    std::copy_n(bytes.begin(), count,
                output.begin() + static_cast<std::ptrdiff_t>(received));
    received += count;
    if (!check((*client)->consume_received(count).has_value(),
               "acknowledge slow client input"))
      return false;
    if (received == input.size() && uart.pending_input() == 0 &&
        uart.pending() == 0 && server->pending_output() == 0)
      break;
  }
  if (!check(rx_blocked && tx_blocked && sent == input.size() &&
                 received == input.size() && input == output &&
                 state.registers[5] == input.size(),
             "native guest echo preserves binary order through RX/TX "
             "backpressure"))
    return false;
  if (!check((*client)->close().has_value(), "close real guest input"))
    return false;
  for (unsigned i = 0; i < 100 && !server->input_closed(); i++)
    if (!check(server->poll().has_value(), "observe completed guest EOF"))
      return false;
  return check(server->input_closed() && server->close().has_value(),
               "settle guest disconnect");
}
