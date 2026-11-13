#include <cstdint>
#include <cstdlib>
#include <print>
#include <string_view>
#include <dynamips/macros.h>
import jit.memory;
import jit.mips64;
import jit.mmu;

namespace {
namespace mem = jit::memory;
namespace cpu = jit::mips64;
fn require(bool condition, std::string_view message) -> void {
  if (!condition) {
    std::println(stderr, "MMU test: {}", message);
    std::exit(1);
  }
}
fn translate(const cpu::cop0_state &state, std::uint64_t address,
               mem::access_kind access = mem::access_kind::read,
               mem::access_size size = mem::access_size::word) {
  return jit::mmu::translate(state, {address}, access, size);
}
fn fault(const cpu::cop0_state &state, std::uint64_t address,
           mem::fault_reason reason,
           mem::access_kind access = mem::access_kind::read) -> void {
  const let value = translate(state, address, access);
  require(!value && value.error().reason == reason &&
              value.error().address.value == address &&
              value.error().access == access &&
              value.error().size == mem::access_size::word,
          "translation retains exact architectural fault information");
}
fn segments() -> void {
  cpu::cop0_state state{};
  for (const std::uint64_t address :
       {0xffffffff80001234ULL, 0xffffffffa0001234ULL}) {
    const let result = translate(state, address);
    require(result && result->value == 0x1234,
            "kernel compatibility aliases translate without TLB");
  }
  fault(state, 0x80001234, mem::fault_reason::address_error);
  fault(state, 0xffffffff80000001ULL, mem::fault_reason::address_error);
  fault(state, 0x1234, mem::fault_reason::tlb_miss, mem::access_kind::execute);
  state.registers[12] = 0x10;
  fault(state, 0xffffffff80001234ULL, mem::fault_reason::address_error);
  fault(state, 0xffffffffc0001234ULL, mem::fault_reason::address_error);
  state.registers[12] = 8;
  fault(state, 0xffffffffc0001234ULL, mem::fault_reason::tlb_miss);
  fault(state, 0xffffffffe0001234ULL, mem::fault_reason::address_error);
  fault(state, 0xffffffffa0001234ULL, mem::fault_reason::address_error);
  state.registers[12] = 0x12;
  require(translate(state, 0xffffffff80001234ULL).has_value(),
          "EXL overrides user privilege");
  state.registers[12] = 0x14;
  require(translate(state, 0x1234) && translate(state, 0x1234)->value == 0x1234,
          "ERL overrides privilege and bypasses the low-2GiB TLB");
  state.registers[12] = 0xe0;
  for (const std::uint64_t cca : {0ULL, 3ULL, 7ULL}) {
    const let result =
        translate(state, 0x8000000000000000ULL | (cca << 59) | 0xffffffffcULL);
    require(result && result->value == 0xffffffffcULL,
            "XKPHYS retains all 36 physical bits and ignores CCA");
  }
  fault(state, 0x8000001000000000ULL, mem::fault_reason::address_error);
  fault(state, 0x0000010000000000ULL, mem::fault_reason::address_error);
  fault(state, 0x4000010000000000ULL, mem::fault_reason::address_error);
  fault(state, 0xc000010000000000ULL, mem::fault_reason::address_error);
  fault(state, 0x000000fffffff000ULL, mem::fault_reason::tlb_miss);
  fault(state, 0x400000fffffff000ULL, mem::fault_reason::tlb_miss);
  fault(state, 0xc00000ff7ffff000ULL, mem::fault_reason::tlb_miss);
  fault(state, 0xc00000ff80000000ULL, mem::fault_reason::address_error);
  state.registers[12] = 0x30;
  fault(state, 0x000000fffffff000ULL, mem::fault_reason::tlb_miss);
  fault(state, 0x4000000000001000ULL, mem::fault_reason::address_error);
  fault(state, 0x8000000000001000ULL, mem::fault_reason::address_error);
  state.registers[12] = 0x48;
  fault(state, 0x4000000000001000ULL, mem::fault_reason::tlb_miss);
  fault(state, 0xc000000000001000ULL, mem::fault_reason::address_error);
  state.registers[12] = 0x80;
  fault(state, 0x100000000ULL, mem::fault_reason::address_error);
  state.registers[12] = 0x20;
  fault(state, 0x100000000ULL, mem::fault_reason::tlb_miss);
  state.registers[12] = 0x28;
  fault(state, 0x100000000ULL, mem::fault_reason::tlb_miss);
  fault(state, 0x4000000100000000ULL, mem::fault_reason::address_error);
}
fn tlb_causes_and_sizes() -> void {
  cpu::cop0_state state{};
  state.registers[10] = 7;
  require(state.translations.write(0, {0, 0x4007, {0x42, 0x82}}).has_value(),
          "install valid clean ASID mapping");
  const let clean = translate(state, 0x4010);
  require(clean && clean->value == 0x1010, "CPU MMU selects the matching ASID");
  fault(state, 0x4010, mem::fault_reason::tlb_modified,
        mem::access_kind::write);
  state.registers[10] = 8;
  fault(state, 0x4010, mem::fault_reason::tlb_miss);
  require(state.translations.write(0, {0, 0x4007, {0x41, 0x81}}).has_value(),
          "install globally matching invalid entries");
  fault(state, 0x4010, mem::fault_reason::tlb_invalid);
  for (const std::uint64_t mask : {0ULL, 0x6000ULL, 0x1e000ULL, 0x7e000ULL,
                                   0x1fe000ULL, 0x7fe000ULL, 0x1ffe000ULL}) {
    const let bytes = ((mask | 0x1fffULL) + 1) / 2;
    const let even = 0x10000000ULL;
    const let odd = 0x18000000ULL;
    require(state.translations
                .write(0, {mask, 7, {(even >> 6) | 6, (odd >> 6) | 6}})
                .has_value(),
            "install supported page-size pair");
    state.registers[10] = 7;
    const let first = translate(state, bytes - 4, mem::access_kind::write);
    const let second = translate(state, bytes + 12);
    require(first && first->value == even + bytes - 4 && second &&
                second->value == odd + 12,
            "MMU preserves even/odd page selection and offsets at every "
            "supported size");
  }
}
} // namespace

fn main() -> int {
  segments();
  tlb_causes_and_sizes();
  std::println("MMU: privilege, segments, ASIDs, TLB causes and page-size "
               "checks passed");
}
