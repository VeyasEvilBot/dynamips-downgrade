module;

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>

#include <dynamips/macros.h>

export module machine.interrupts;

import jit.devices;
import jit.mips64;

export namespace dynamips::machine {
enum class interrupt_error { invalid_line, capacity_exhausted };

// One execution owner. CP0 outlives the router; peripherals disconnect their
// borrowed source handles before the router is destroyed.
class interrupt_router {
public:
  static constexpr std::size_t capacity = 16;

  explicit interrupt_router(jit::mips64::cop0_state &cpu) noexcept
      : cpu_(cpu) {}

  interrupt_router(const interrupt_router &) = delete;
  fn operator=(const interrupt_router &) -> interrupt_router & = delete;
  interrupt_router(interrupt_router &&) = delete;
  fn operator=(interrupt_router &&) -> interrupt_router & = delete;

  ~interrupt_router() { reset(); }

  [[nodiscard]] fn connect(std::uint8_t line) noexcept
      -> std::expected<jit::devices::interrupt_line, interrupt_error> {
    if (line < 2 || line > 6)
      return std::unexpected(interrupt_error::invalid_line);
    if (count_ == sources_.size())
      return std::unexpected(interrupt_error::capacity_exhausted);
    const let mask = std::uint64_t{1} << (8u + line);
    let &entry = sources_[count_];
    entry = {this, mask, false};
    count_++;
    owned_mask_ |= mask;
    refresh();
    return jit::devices::interrupt_line{&entry, publish};
  }

  // Reset electrical levels while retaining the registered wiring.
  fn reset() noexcept -> void {
    for (std::size_t i = 0; i < count_; i++)
      sources_[i].asserted = false;
    refresh();
  }

private:
  struct source {
    interrupt_router *owner = nullptr;
    std::uint64_t mask = 0;
    bool asserted = false;
  };

  jit::mips64::cop0_state &cpu_;
  std::array<source, capacity> sources_{};
  std::size_t count_ = 0;
  std::uint64_t owned_mask_ = 0;

  fn refresh() noexcept -> void {
    std::uint64_t asserted = 0;
    for (std::size_t i = 0; i < count_; i++)
      if (sources_[i].asserted)
        asserted |= sources_[i].mask;
    cpu_.registers[13] = (cpu_.registers[13] & ~owned_mask_) | asserted;
  }

  static fn publish(void *context, bool asserted) noexcept -> void {
    let &entry = *static_cast<source *>(context);
    entry.asserted = asserted;
    entry.owner->refresh();
  }
};
} // namespace dynamips::machine
