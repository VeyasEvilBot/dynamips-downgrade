module;
#include <cstdint>
#include <expected>
#include <dynamips/macros.h>
export module jit.mmu;
import jit.memory;
import jit.mips64;
import jit.tlb;

export namespace jit::mmu {
inline constexpr std::uint64_t physical_size = std::uint64_t{1} << 36;
enum class privilege : std::uint8_t { kernel, supervisor, user };

constexpr fn current_privilege(std::uint64_t status) noexcept -> privilege {
  if (status & 6)
    return privilege::kernel;
  switch ((status >> 3) & 3) {
  case 0:
    return privilege::kernel;
  case 1:
    return privilege::supervisor;
  default:
    return privilege::user;
  }
}

constexpr fn extended_addressing(std::uint64_t status,
                                  memory::virtual_address address) noexcept
    -> bool {
  const let region = address.value >> 62;
  const unsigned bit = region == 0 ? 5u : region == 1 ? 6u : 7u;
  return (status & (std::uint64_t{1} << bit)) != 0;
}

// The implemented CPU uses 40-bit mapped virtual regions and 36-bit physical
// addresses. Cacheability attributes do not change the physical bus address.
[[nodiscard]] fn
translate(const mips64::cop0_state &cpu, memory::virtual_address address,
          memory::access_kind access, memory::access_size size) noexcept
    -> std::expected<memory::physical_address, memory::fault> {
  const let failure = [&](memory::fault_reason reason) {
    return std::unexpected(memory::fault{address, reason, access, size});
  };
  const let bytes = static_cast<unsigned>(size);
  if (bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8)
    return failure(memory::fault_reason::unsupported_size);
  if (address.value % bytes)
    return failure(memory::fault_reason::address_error);
  const let status = cpu.registers[12];
  const let mode = current_privilege(status);
  const bool kernel = mode == privilege::kernel;
  const bool supervisor = mode != privilege::user;
  const bool extended = extended_addressing(status, address);
  const let va = address.value;
  bool mapped = false;
  if (va < 0x80000000ULL) {
    if (status & 4)
      return memory::physical_address{va};
    mapped = true;
  } else if (va >= 0xffffffff80000000ULL) {
    if (va < 0xffffffffc0000000ULL) {
      if (!kernel)
        return failure(memory::fault_reason::address_error);
      return memory::physical_address{va & 0x1fffffffULL};
    }
    if (va < 0xffffffffe0000000ULL ? !supervisor : !kernel)
      return failure(memory::fault_reason::address_error);
    mapped = true;
  } else if (extended) {
    const let region = va >> 62;
    constexpr std::uint64_t offset_mask = (std::uint64_t{1} << 40) - 1;
    if (region == 2) {
      constexpr std::uint64_t physical_mask = physical_size - 1;
      if (!kernel || (va & 0x07fffff000000000ULL))
        return failure(memory::fault_reason::address_error);
      return memory::physical_address{va & physical_mask};
    }
    if ((va & 0x3fffffffffffffffULL) > offset_mask ||
        (region == 3 && va > 0xc00000ff7fffffffULL) ||
        (region == 1 && !supervisor) || (region == 3 && !kernel))
      return failure(memory::fault_reason::address_error);
    mapped = true;
  }
  if (!mapped)
    return failure(memory::fault_reason::address_error);
  const let physical = cpu.translations.translate(
      address, static_cast<std::uint8_t>(cpu.registers[10]), access);
  if (physical)
    return *physical;
  switch (physical.error()) {
  case tlb::error::miss:
    return failure(memory::fault_reason::tlb_miss);
  case tlb::error::invalid:
    return failure(memory::fault_reason::tlb_invalid);
  case tlb::error::modified:
    return failure(memory::fault_reason::tlb_modified);
  default:
    return failure(memory::fault_reason::bus);
  }
}
} // namespace jit::mmu
