#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <print>
#include <string_view>

#include <dynamips/macros.h>

import jit.exceptions;
import jit.memory;
import jit.mips64;

namespace {
namespace mem = jit::memory;
namespace cpu = jit::mips64;
namespace traps = jit::exceptions;

fn require(bool condition, std::string_view message) -> void {
  if (!condition) {
    std::println(stderr, "exception test: {}", message);
    std::exit(1);
  }
}

fn classification() -> void {
  for (const let access : {mem::access_kind::read, mem::access_kind::write,
                           mem::access_kind::execute}) {
    const bool store = access == mem::access_kind::write;
    for (const let reason :
         {mem::fault_reason::tlb_miss, mem::fault_reason::tlb_invalid,
          mem::fault_reason::tlb_modified, mem::fault_reason::misaligned,
          mem::fault_reason::address_error}) {
      const let issue = traps::from_memory(
          {{0x12345678}, reason, access, mem::access_size::word}, 0x8004,
          0x8000);
      const bool tlb = reason == mem::fault_reason::tlb_miss ||
                       reason == mem::fault_reason::tlb_invalid ||
                       reason == mem::fault_reason::tlb_modified;
      const std::uint32_t code = reason == mem::fault_reason::tlb_modified ? 1u
                                 : tlb ? (store ? 3u : 2u)
                                       : (store ? 5u : 4u);
      require(
          issue && issue->code == code && issue->pc == 0x8004 &&
              issue->branch_pc == 0x8000 && issue->bad_vaddr == 0x12345678 &&
              issue->tlb == tlb &&
              issue->refill == (reason == mem::fault_reason::tlb_miss),
          "memory access classifications preserve architectural trap metadata");
    }
  }
  for (const let reason :
       {mem::fault_reason::unmapped, mem::fault_reason::permission,
        mem::fault_reason::unsupported_size, mem::fault_reason::bus})
    require(!traps::from_memory(
                {{0}, reason, mem::access_kind::read, mem::access_size::word},
                0x8000),
            "unmodeled hardware and host-map errors remain emulator failures");
}

fn exception_entry() -> void {
  std::array<std::byte, mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 1);
  require(memory &&
              (*memory)
                  ->map_ram({0x1000}, {0},
                            mem::permission::read | mem::permission::write)
                  .has_value(),
          "create exception reservation backing");
  for (const bool bev : {false, true}) {
    for (const bool nested : {false, true}) {
      for (const bool delay : {false, true}) {
        cpu::state state{};
        state.cop0.registers[12] = (bev ? 1u << 22 : 0u) | (nested ? 2u : 0u);
        state.cop0.registers[13] = 0x80008000;
        state.cop0.registers[14] = 0xdeadbeef;
        state.cop0.registers[4] = 0x1234567800000000ULL;
        state.cop0.registers[20] = 0x9876543200000000ULL;
        state.cop0.registers[10] = 0xa5;
        require((*memory)->load_linked<std::uint32_t>({0x1000}).has_value(),
                "reserve before exception");
        const let issue = traps::from_memory(
            {{0xc000001234567890ULL},
             mem::fault_reason::tlb_miss,
             mem::access_kind::read,
             mem::access_size::word},
            0x8004,
            delay ? std::optional<std::uint64_t>{0x8000} : std::nullopt);
        require(issue.has_value(), "classify refill");
        traps::enter(state, **memory, *issue);
        const std::uint64_t vector =
            nested ? (bev ? 0xffffffffbfc00380ULL : 0xffffffff80000180ULL)
                   : (bev ? 0xffffffffbfc00200ULL : 0xffffffff80000000ULL);
        require(state.pc.value == vector && (state.cop0.registers[12] & 2) &&
                    state.cop0.registers[8] == issue->bad_vaddr &&
                    (state.cop0.registers[13] & 0x807c) == 0x8008 &&
                    state.cop0.registers[14] == (nested  ? 0xdeadbeefu
                                                 : delay ? 0x8000u
                                                         : 0x8004u) &&
                    static_cast<bool>(state.cop0.registers[13] & 0x80000000) ==
                        (nested || delay),
                "refill entry preserves pending interrupts and nested EPC/BD");
        require(state.cop0.registers[4] == 0x12345678001a2b30ULL &&
                    state.cop0.registers[20] == 0x98765433891a2b30ULL &&
                    state.cop0.registers[10] == 0xc0000012345660a5ULL,
                "TLB entry fills Context, XContext and EntryHi while "
                "preserving page-table bases and ASID");
        require((*memory)->store_conditional<std::uint32_t>({0x1000}, 9) ==
                    false,
                "exception entry clears the linked reservation");
      }
    }
  }
  for (const let reason :
       {mem::fault_reason::tlb_invalid, mem::fault_reason::tlb_modified,
        mem::fault_reason::address_error}) {
    cpu::state state{};
    state.cop0.registers[4] = 0x1234;
    state.cop0.registers[20] = 0x5678;
    state.cop0.registers[10] = 0x9abc;
    const let issue = traps::from_memory(
        {{0x1234}, reason, mem::access_kind::write, mem::access_size::word},
        0x8000);
    require(issue.has_value(), "classify general memory exception");
    traps::enter(state, **memory, *issue);
    require(state.pc.value == 0xffffffff80000180ULL,
            "invalid, modified and address exceptions use general vector");
    if (reason == mem::fault_reason::address_error)
      require(state.cop0.registers[4] == 0x1234 &&
                  state.cop0.registers[20] == 0x5678 &&
                  state.cop0.registers[10] == 0x9abc,
              "address errors preserve TLB context registers");
  }
}

fn vector_width() -> void {
  std::array<std::byte, 0> ram{};
  let memory = mem::address_space::create(ram, 0);
  require(memory.has_value(), "create vector test address space");
  for (const let address :
       {0x1234000ULL, 0x4000000001234000ULL, 0xc000000001234000ULL,
        0xffffffffc1234000ULL, 0xffffffffe1234000ULL}) {
    const unsigned width_bit = address < 0x4000000000000000ULL ? 5u
                               : address < 0x8000000000000000ULL ||
                                       (address >= 0xffffffffc0000000ULL &&
                                        address < 0xffffffffe0000000ULL)
                                   ? 6u
                                   : 7u;
    for (const unsigned enabled_bit : {5u, 6u, 7u}) {
      cpu::state state{};
      state.cop0.registers[12] = std::uint64_t{1} << enabled_bit;
      const let issue = traps::from_memory({{address},
                                            mem::fault_reason::tlb_miss,
                                            mem::access_kind::read,
                                            mem::access_size::word},
                                           0x8000);
      require(issue.has_value(), "classify vector-width refill");
      traps::enter(state, **memory, *issue);
      require(state.pc.value == 0xffffffff80000000ULL +
                                    (width_bit == enabled_bit ? 0x80u : 0u),
              "refill width follows the faulting address-space bit before EXL "
              "forces kernel mode");
    }
  }
}
} // namespace

fn main() -> int {
  classification();
  exception_entry();
  vector_width();
  std::println("exceptions: classification, refill vectors, CP0 state, nested "
               "delay slots and reservations passed");
}
