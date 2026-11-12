module;
#include <cstdint>
#include <span>
#include <dynamips/macros.h>
export module devices.serial_eeprom;

export namespace dynamips::devices {
// Read-only NMC93C46, 64 words. Borrowed board data outlives the device.
class nmc93c46 {
public:
  explicit nmc93c46(std::span<const std::uint16_t> words = {}) noexcept
      : words_(words) {}

  fn reset() noexcept -> void {
    selected_ = clock_ = false;
    output_ = true;
    phase_ = phase::inactive;
    command_ = 0;
    position_ = 0;
  }
  fn output() const noexcept -> bool { return output_; }
  fn write(bool selected, bool clock, bool data) noexcept -> void {
    output_ = true;
    if (selected != selected_) {
      phase_ = selected ? phase::command : phase::inactive;
      command_ = 0;
      position_ = 0;
    }
    if (selected && clock && !clock_) {
      if (phase_ == phase::command) {
        if (position_ || data) {
          command_ |= static_cast<std::uint16_t>(static_cast<unsigned>(data)
                                                 << position_);
          position_++;
          if (position_ == 9) {
            phase_ = (command_ & 7) == 3 ? phase::read : phase::inactive;
            position_ = 0;
          }
        }
      } else if (phase_ == phase::read) {
        const let address = (command_ >> 3) & 63u;
        const std::uint16_t word =
            address < words_.size() ? words_[address] : 0xffff;
        output_ = (word & (1u << position_)) != 0;
        position_++;
        if (position_ == 16)
          phase_ = phase::inactive;
      }
    }
    selected_ = selected;
    clock_ = clock;
  }

private:
  enum class phase : std::uint8_t { inactive, command, read };
  std::span<const std::uint16_t> words_;
  std::uint16_t command_ = 0;
  std::uint8_t position_ = 0;
  phase phase_ = phase::inactive;
  bool selected_ = false, clock_ = false, output_ = true;
};
} // namespace dynamips::devices
