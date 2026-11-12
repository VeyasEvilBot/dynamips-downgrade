module;
#include <bit>
#include <cstdint>
#include <expected>
#include <dynamips/macros.h>
export module devices.c7200_iofpga;
import jit.devices;
import jit.memory;

export namespace dynamips::devices {
class c7200_iofpga {
public:
  c7200_iofpga() noexcept : endpoint_{this, read, write} {}

  c7200_iofpga(const c7200_iofpga &) = delete;
  fn operator=(const c7200_iofpga &) -> c7200_iofpga & = delete;

  fn descriptor() noexcept -> jit::devices::descriptor {
    return {&endpoint_, jit::devices::widths::half, std::endian::big};
  }

private:
  jit::memory::device endpoint_;

  static fn read(void *, jit::memory::device_access access) noexcept
      -> std::expected<std::uint64_t, jit::memory::device_error> {
    if (access.size != jit::memory::access_size::half)
      return std::unexpected(jit::memory::device_error::unsupported_size);
    // This board revision probe reads zero in the legacy NPE-400 model.
    if (access.address.value == 0x1e8403a0)
      return 0;
    return std::unexpected(jit::memory::device_error::bus);
  }

  static fn write(void *context, jit::memory::device_access access,
                   std::uint64_t) noexcept
      -> std::expected<void, jit::memory::device_error> {
    const let result = read(context, access);
    if (!result)
      return std::unexpected(result.error());
    return {};
  }
};
} // namespace dynamips::devices
