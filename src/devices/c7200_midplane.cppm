module;
#include <bit>
#include <cstdint>
#include <expected>
#include <dynamips/macros.h>
export module devices.c7200_midplane;
import jit.devices;
import jit.memory;

export namespace dynamips::devices {
class c7200_midplane {
public:
  c7200_midplane() noexcept : endpoint_{this, read, write} {}

  c7200_midplane(const c7200_midplane &) = delete;
  fn operator=(const c7200_midplane &) -> c7200_midplane & = delete;

  fn descriptor() noexcept -> jit::devices::descriptor {
    return {&endpoint_, jit::devices::widths::byte, std::endian::big};
  }

private:
  jit::memory::device endpoint_;

  static fn read(void *, jit::memory::device_access access) noexcept
      -> std::expected<std::uint64_t, jit::memory::device_error> {
    if (access.size != jit::memory::access_size::byte)
      return std::unexpected(jit::memory::device_error::unsupported_size);
    // The legacy FPGA ignores the frequently accessed priority register.
    if (access.address.value == 0x1e80007b)
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
