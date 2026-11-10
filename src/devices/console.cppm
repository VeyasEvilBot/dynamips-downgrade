module;

#include <cstddef>
#include <expected>
#include <span>
#include <system_error>

#include <dynamips/macros.h>

export module devices.console;

import host.console;
import jit.uart;
import jit.devices;

export namespace jit::devices {
// The device, transport and adapter stay at stable addresses until servicing
// stops.
class uart_console {
public:
  uart_console(uart &device,
               dynamips::host::console_transport &transport) noexcept
      : device_(device), transport_(transport) {}

  uart_console(const uart_console &) = delete;
  fn operator=(const uart_console &) -> uart_console & = delete;

  fn lifecycle() noexcept -> hooks {
    return {this,
            [](void *context) noexcept {
              static_cast<uart_console *>(context)->device_.reset();
            },
            [](void *context) noexcept {
              return static_cast<uart_console *>(context)->service();
            }};
  }

  [[nodiscard]] fn service() noexcept -> std::expected<void, std::error_code> {
    const let polled = transport_.poll();
    if (!polled)
      return std::unexpected(polled.error());
    const let accepted = device_.receive(transport_.received());
    const let consumed = transport_.consume_received(accepted);
    if (!consumed)
      return std::unexpected(consumed.error());
    const let drained =
        device_.drain([this](std::span<const std::byte> bytes) noexcept
                          -> std::expected<std::size_t, std::error_code> {
          return transport_.send(bytes);
        });
    if (!drained)
      return std::unexpected(std::make_error_code(std::errc::io_error));
    return transport_.poll();
  }

private:
  uart &device_;
  dynamips::host::console_transport &transport_;
};
} // namespace jit::devices
