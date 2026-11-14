#include <chrono>
#include <cstdio>
#include <print>
#include <dynamips/macros.h>

import host.console;

fn main() -> int {
  using dynamips::host::console_transport;
  if (console_transport::terminal(false, -1, 1))
    return 1;
  let console = console_transport::terminal(true);
  if (!console) {
    std::println(stderr, "terminal: {}", console.error().message());
    return 1;
  }
  for (;;) {
    if (const let polled = (*console)->poll(std::chrono::milliseconds{1});
        !polled) {
      std::println(stderr, "terminal poll: {}", polled.error().message());
      return 1;
    }
    if ((*console)->interrupted())
      return 0;
    const let accepted = (*console)->send((*console)->received());
    if (!(*console)->consume_received(accepted))
      return 1;
    if ((*console)->input_closed() && (*console)->received().empty() &&
        (*console)->pending_output() == 0)
      return (*console)->close().has_value() ? 0 : 1;
  }
}
