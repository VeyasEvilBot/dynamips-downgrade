module;

#include <cstdint>
#include <optional>

#include <dynamips/macros.h>

export module jit.exceptions;

import jit.memory;
import jit.mips64;

export namespace jit::exceptions {
struct descriptor {
  std::uint32_t code;
  std::uint64_t pc;
  std::optional<std::uint64_t> branch_pc;
  std::uint64_t bad_vaddr;
  bool tlb;
  bool refill;
};

[[nodiscard]] fn
from_memory(memory::fault fault, std::uint64_t pc,
            std::optional<std::uint64_t> branch_pc = {}) noexcept
    -> std::optional<descriptor> {
  const bool store = fault.access == memory::access_kind::write;
  switch (fault.reason) {
  case memory::fault_reason::misaligned:
  case memory::fault_reason::address_error:
    return descriptor{store ? 5u : 4u,     pc,    branch_pc,
                      fault.address.value, false, false};
  case memory::fault_reason::tlb_miss:
    return descriptor{store ? 3u : 2u,     pc,   branch_pc,
                      fault.address.value, true, true};
  case memory::fault_reason::tlb_invalid:
    return descriptor{store ? 3u : 2u,     pc,   branch_pc,
                      fault.address.value, true, false};
  case memory::fault_reason::tlb_modified:
    return descriptor{1, pc, branch_pc, fault.address.value, true, false};
  default:
    return std::nullopt;
  }
}

// Refill vector width follows the referenced address space, before EXL changes.
[[nodiscard]] fn extended_refill(std::uint64_t address,
                                  std::uint64_t status) noexcept -> bool {
  unsigned bit = 7;
  if (address < 0x4000000000000000ULL)
    bit = 5;
  else if (address < 0x8000000000000000ULL ||
           (address >= 0xffffffffc0000000ULL &&
            address < 0xffffffffe0000000ULL))
    bit = 6;
  return (status & (std::uint64_t{1} << bit)) != 0;
}

// Execution has returned and released its instruction/write monitor.
fn enter(mips64::state &state, memory::address_space &memory,
           const descriptor &issue) noexcept -> void {
  let &registers = state.cop0.registers;
  const let status = registers[12];
  const bool refill = issue.refill && !(status & 2);
  registers[8] = issue.bad_vaddr;
  if (issue.tlb) {
    registers[4] = (registers[4] & ~std::uint64_t{0x7fffff}) |
                   ((issue.bad_vaddr >> 9) & 0x7ffff0);
    registers[20] = (registers[20] & ~std::uint64_t{0x1ffffffff}) |
                    ((issue.bad_vaddr >> 9) & 0x7ffffff0) |
                    ((issue.bad_vaddr >> 31) & 0x180000000);
    registers[10] =
        (registers[10] & 255) | (issue.bad_vaddr & 0xc00000ffffffe000ULL);
  }
  if (!(status & 2))
    memory.invalidate();
  else
    memory.clear_reservation();
  state.pc =
      state.cop0.exception(issue.code, {issue.branch_pc.value_or(issue.pc)},
                           issue.branch_pc.has_value());
  if (refill) {
    const std::uint64_t base =
        status & (1u << 22) ? 0xffffffffbfc00200ULL : 0xffffffff80000000ULL;
    state.pc = {base + (extended_refill(issue.bad_vaddr, status) ? 0x80u : 0u)};
  }
}
} // namespace jit::exceptions
