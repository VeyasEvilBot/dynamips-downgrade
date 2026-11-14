module;
#include <cstdint>
#include <expected>
#include <dynamips/macros.h>
export module devices.c7200_environment;

export namespace dynamips::devices {
enum class adc_error { unsupported_channel };

// Raw nominal inputs use the NPE-400 calibration in dev_c7200_iofpga.c.
// The execution owner may change inputs while stopped to model rail faults.
struct c7200_environment {
  std::uint8_t power_enabled = 3;
  std::uint8_t power0 = 127, power1 = 127;
  std::uint8_t positive_3v3 = 127, positive_12 = 127;
  std::uint8_t positive_5 = 127, negative_12 = 127;

  [[nodiscard]] fn sample(std::uint32_t selector) const noexcept
      -> std::expected<std::uint8_t, adc_error> {
    switch (selector) {
    case 0:
      return power_enabled & 1 ? power0 : std::uint8_t{0};
    case 2:
      return power_enabled & 2 ? power1 : std::uint8_t{0};
    case 4:
      return positive_3v3;
    case 8:
      return positive_12;
    case 10:
      return positive_5;
    case 12:
      return negative_12;
    default:
      return std::unexpected(adc_error::unsupported_channel);
    }
  }
};
} // namespace dynamips::devices
