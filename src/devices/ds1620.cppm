module;
#include <cstdint>
#include <expected>
#include <dynamips/macros.h>
export module devices.ds1620;

export namespace dynamips::devices {
enum class temperature_error { unsupported_command };

// Fixed 22 C ambient. Conversion and nonvolatile writes complete synchronously;
// serial edges, register widths and status behavior follow the DS1620 protocol.
class ds1620 {
public:
  fn reset() noexcept -> void {
    selected_ = clock_ = input_ = false;
    output_ = true;
    phase_ = phase::inactive;
    shift_ = position_ = 0;
  }
  fn input(bool bit) noexcept -> void { input_ = bit; }
  fn output() const noexcept -> bool { return clock_ || output_; }

  [[nodiscard]] fn pins(bool selected, bool clock) noexcept
      -> std::expected<void, temperature_error> {
    if (!selected) {
      const let input = input_;
      reset();
      input_ = input;
      clock_ = clock;
      return {};
    }
    if (!selected_) {
      phase_ = phase::command;
      shift_ = position_ = 0;
      output_ = true;
    }
    const bool rising = clock && !clock_;
    const bool falling = !clock && clock_;
    selected_ = selected;
    clock_ = clock;
    if (rising && (phase_ == phase::command || phase_ == phase::write)) {
      shift_ |= static_cast<std::uint16_t>(static_cast<unsigned>(input_)
                                           << position_);
      position_++;
      if (phase_ == phase::command && position_ == 8)
        return command(static_cast<std::uint8_t>(shift_));
      if (phase_ == phase::write && position_ == width_) {
        if (command_ == 1)
          high_ = shift_ & 0x1ff;
        else if (command_ == 2)
          low_ = shift_ & 0x1ff;
        else
          config_ = static_cast<std::uint8_t>(
              (config_ & 0x80) | (config_ & shift_ & 0x60) | 8 | (shift_ & 3));
        phase_ = phase::inactive;
      }
    } else if (falling && phase_ == phase::read) {
      output_ = position_ < width_ && ((shift_ >> position_) & 1);
      if (position_ < width_)
        position_++;
    }
    return {};
  }

private:
  enum class phase : std::uint8_t { inactive, command, read, write };
  phase phase_ = phase::inactive;
  std::uint16_t high_ = 0, low_ = 0, shift_ = 0;
  std::uint8_t config_ = 0x8a, command_ = 0, position_ = 0, width_ = 0;
  bool selected_ = false, clock_ = false, input_ = false, output_ = true;
  static constexpr std::uint16_t temperature = 44;

  static fn signed_temperature(std::uint16_t value) noexcept -> int {
    return (value & 0x100) ? static_cast<int>(value) - 512 : value;
  }

  fn command(std::uint8_t value) noexcept
      -> std::expected<void, temperature_error> {
    command_ = value;
    shift_ = position_ = 0;
    width_ = 9;
    phase_ = phase::read;
    switch (value) {
    case 0xaa:
      shift_ = temperature;
      break;
    case 0xa0:
      shift_ = 12;
      break;
    case 0xa9:
      shift_ = 16;
      break;
    case 0xa1:
      shift_ = high_;
      break;
    case 0xa2:
      shift_ = low_;
      break;
    case 0xac:
      shift_ = config_;
      width_ = 8;
      break;
    case 1:
    case 2:
    case 0x0c:
      phase_ = phase::write;
      if (value == 0x0c)
        width_ = 8;
      break;
    case 0xee:
      if (signed_temperature(temperature) >= signed_temperature(high_))
        config_ |= 0x40;
      if (signed_temperature(temperature) <= signed_temperature(low_))
        config_ |= 0x20;
      phase_ = phase::inactive;
      break;
    case 0x22:
      phase_ = phase::inactive;
      break;
    default:
      phase_ = phase::inactive;
      return std::unexpected(temperature_error::unsupported_command);
    }
    return {};
  }
};
} // namespace dynamips::devices
