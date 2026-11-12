module;

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <variant>

#include <dynamips/macros.h>

export module jit.execution;

import jit.devices;
import jit.dispatch;
import jit.mips64;

export namespace jit {
struct execution_error {
  std::variant<dispatch_error, devices::service_error> cause;
  std::optional<devices::service_error> service_after_fault{};
};

// Borrowed components outlive the loop. Hooks perform bounded work; transports
// can enqueue or poll I/O and leave blocked device queues pending. Service
// errors stop the run; completed guest counters are returned only on success.
template <typename Dispatcher = dispatcher> class execution_loop {
public:
  execution_loop(Dispatcher &guest, devices::bus &devices) noexcept
      : guest_(guest), devices_(devices) {}

  [[nodiscard]] fn run_slice(mips64::state &state, std::uint64_t budget,
                              std::size_t block_limit = 64,
                              bool dump_ir = false) noexcept
      -> std::expected<dispatch_result, execution_error> {
    const let before = devices_.service();
    if (!before)
      return std::unexpected(execution_error{before.error()});
    const let executed = guest_.run_slice(state, budget, block_limit, dump_ir);
    const let after = devices_.service();
    if (!executed)
      return std::unexpected(
          execution_error{executed.error(),
                          after ? std::nullopt : std::optional{after.error()}});
    if (!after)
      return std::unexpected(execution_error{after.error()});
    return *executed;
  }

private:
  Dispatcher &guest_;
  devices::bus &devices_;
};
} // namespace jit
