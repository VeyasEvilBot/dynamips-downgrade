#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <print>
#include <memory>
#include <utility>
#include <span>
#include <string_view>
#include <dynamips/macros.h>
import jit.memory;
import jit.mips64;
import jit.mmu;
import jit.physical;
import jit.memory_dispatch;
import jit.dispatch;

namespace {
namespace mem = jit::memory;
namespace cpu = jit::mips64;
constexpr std::uint64_t kernel = 0xffffffff80000000ULL;
fn require(bool ok, std::string_view what) -> void {
  if (!ok) {
    std::println(stderr, "CPU fault test: {}", what);
    std::exit(1);
  }
}

struct fixture {
  std::array<std::byte, 4 * mem::page_size> ram{};
  cpu::state state{};
  std::unique_ptr<mem::address_space> memory;
  jit::physical::bus bus;

  fixture() : bus(make_bus()) {
    let space = mem::address_space::create(ram, 0);
    require(space.has_value(), "create translated address space");
    memory = std::move(*space);
    memory->set_translation(
        this,
        [](void *context, mem::virtual_address va, mem::access_kind access,
           mem::access_size size) noexcept
            -> std::expected<mem::translated_page, mem::fault> {
          let &self = *static_cast<fixture *>(context);
          const let pa = jit::mmu::translate(self.state.cop0, va, access, size);
          if (!pa)
            return std::unexpected(pa.error());
          let result = self.bus.resolve(*pa, access, size);
          if (!result) {
            let fault = result.error();
            fault.address = va;
            return std::unexpected(fault);
          }
          return *result;
        });
  }

  fn make_bus() -> jit::physical::bus {
    const std::array regions{
        jit::physical::region{{0},
                              ram.size(),
                              ram,
                              nullptr,
                              mem::permission::read | mem::permission::write |
                                  mem::permission::execute}};
    let result = jit::physical::bus::create(regions);
    require(result.has_value(), "create physical bus");
    return *result;
  }
  fn word(std::uint64_t pa, std::uint32_t bits) -> void {
    const let encoded = std::byteswap(bits);
    require(memory->write_physical({pa}, std::as_bytes(std::span{&encoded, 1}))
                .has_value(),
            "write physical program");
  }
};

fn data_faults() -> void {
  for (const unsigned tier : {0u, 1u, 2u}) {
    for (const unsigned kind : {0u, 1u, 2u, 3u}) {
      fixture f;
      const bool store = kind == 2;
      f.word(0, 0x24210001);
      f.word(4, store ? 0xac620000 : 0x8c620000);
      f.word(8, 0);
      f.word(12, 0x1424fffc);
      f.word(16, 0);
      let dispatch =
          jit::memory_dispatcher::create(*f.memory, tier == 1 ? 32 : 0);
      require(dispatch.has_value(), "create execution tier");
      f.state.pc = {kernel};
      f.state.registers[3] = kernel + 0x2000;
      f.state.registers[4] = 100000;
      if (tier == 2)
        require((*dispatch)->run_slice(f.state, 30000, 4096).has_value(),
                "warm native trace");
      f.state.pc = {kernel};
      f.state.registers[1] = 0;
      f.state.registers[2] = 99;
      f.state.registers[3] = kind == 3 ? 0x4001 : 0x4000;
      f.state.cop0.registers[9] = 0;
      f.state.cop0.half_cycle = false;
      if (kind == 1 || kind == 2)
        require(f.state.cop0.translations
                    .write(0, {0, 0x4000, {kind == 1 ? 0x80u : 0x82u, 0}})
                    .has_value(),
                "install invalid or clean data page");
      const let result = (*dispatch)->run_slice(f.state, 100, 1);
      const let expected = kind == 2 ? 1u : kind == 3 ? 4u : 2u;
      require(result && result->reason == jit::stop_reason::guest_exception &&
                  result->instructions == 2 &&
                  f.state.cop0.registers[14] == kernel + 4 &&
                  f.state.cop0.registers[8] == f.state.registers[3] &&
                  ((f.state.cop0.registers[13] >> 2) & 31) == expected &&
                  f.state.registers[1] == 1 && f.state.registers[2] == 99 &&
                  f.state.cop0.registers[9] == 1,
              "all tiers deliver architectural data faults with exact state "
              "and instruction counts");
    }
  }
}
fn data_delay_slots() -> void {
  for (const unsigned tier : {0u, 1u, 2u}) {
    fixture f;
    f.word(0, 0x24210001);
    f.word(4, 0x1424fffe);
    f.word(8, 0x8c620000);
    let dispatch =
        jit::memory_dispatcher::create(*f.memory, tier == 1 ? 32 : 0);
    require(dispatch.has_value(), "create delay-slot execution tier");
    f.state.pc = {kernel};
    f.state.registers[3] = kernel + 0x2000;
    f.state.registers[4] = 100000;
    if (tier == 2)
      require((*dispatch)->run_slice(f.state, 30000, 4096).has_value(),
              "warm delay-slot native trace");
    f.state.pc = {kernel};
    f.state.registers[1] = 0;
    f.state.registers[2] = 99;
    f.state.registers[3] = 0x4000;
    f.state.cop0.registers[9] = 0;
    f.state.cop0.half_cycle = false;
    const let result = (*dispatch)->run_slice(f.state, 100, 1);
    require(result && result->reason == jit::stop_reason::guest_exception &&
                result->instructions == 3 &&
                f.state.cop0.registers[14] == kernel + 4 &&
                (f.state.cop0.registers[13] & 0x80000000u) &&
                f.state.cop0.registers[8] == 0x4000 &&
                f.state.registers[1] == 1 && f.state.registers[2] == 99 &&
                f.state.cop0.registers[9] == 1 && f.state.cop0.half_cycle,
            "all tiers preserve BD, branch EPC and exact delay-fault timing");
  }
}
fn fetch_slots() -> void {
  for (const bool annul : {false, true}) {
    fixture f;
    f.word(0xffc, annul ? 0x50220001 : 0x0c000800);
    require(f.state.cop0.translations.write(0, {0, 0, {6, 0}}).has_value(),
            "map code with invalid delay-slot page");
    f.state.pc = {0xffc};
    f.state.registers[1] = 1;
    f.state.registers[2] = 2;
    let dispatch = jit::memory_dispatcher::create(*f.memory);
    require(dispatch.has_value(), "create delay-fetch dispatcher");
    const let result = (*dispatch)->run_slice(f.state, 100, 1);
    if (annul)
      require(result && result->instructions == 1 &&
                  f.state.pc.value == 0x1004 &&
                  !(f.state.cop0.registers[12] & 2),
              "annulled unfetchable slots do not trap");
    else
      require(result && result->reason == jit::stop_reason::guest_exception &&
                  result->instructions == 2 &&
                  f.state.registers[31] == 0x1004 &&
                  f.state.cop0.registers[14] == 0xffc &&
                  (f.state.cop0.registers[13] & 0x80000000u) &&
                  f.state.cop0.registers[8] == 0x1000,
              "delay-slot fetch faults preserve the branch link, BD and EPC");
  }
}
fn refill_and_privilege() -> void {
  fixture f;
  constexpr std::array handler{0x40800000u, 0x34054000u, 0x40855000u,
                               0x34050086u, 0x40851000u, 0x40801800u,
                               0x42000002u, 0x42000018u};
  for (std::size_t i = 0; i < handler.size(); i++)
    f.word(i * 4, handler[i]);
  f.word(0x1000, 0x8c620000);
  f.word(0x1004, 0x34040001);
  f.word(0x1008, 0x1000ffff);
  f.word(0x100c, 0);
  f.word(0x2000, 0x12345678);
  f.state.pc = {kernel + 0x1000};
  f.state.registers[3] = 0x4000;
  let dispatch = jit::memory_dispatcher::create(*f.memory, 32);
  require(dispatch.has_value(), "create refill execution");
  const let miss = (*dispatch)->run_slice(f.state, 100);
  require(miss && miss->reason == jit::stop_reason::guest_exception &&
              f.state.pc.value == kernel,
          "unmapped load enters guest refill handler");
  const let resumed = (*dispatch)->run_slice(f.state, 100, 16);
  require(
      resumed && f.state.registers[2] == 0x12345678 &&
          f.state.registers[4] == 1 && !(f.state.cop0.registers[12] & 2),
      "guest TLBWI plus ERET retries the original load and resumes execution");
  fixture p;
  p.word(0, 0x3c018000);
  p.word(4, 0x8c220000);
  p.word(8, 0x34030010);
  p.word(12, 0x40836000);
  p.word(16, 0x34040009);
  p.word(0x180, 0x42000018);
  p.state.pc = {kernel};
  let privileged = jit::memory_dispatcher::create(*p.memory);
  require(privileged.has_value(), "create privilege execution");
  require((*privileged)->run_slice(p.state, 10, 1).has_value() &&
              p.state.pc.value == kernel + 16,
          "MTC0 Status ends the kernel block");
  const let denied = (*privileged)->run_slice(p.state, 10, 1);
  require(denied && denied->reason == jit::stop_reason::guest_exception &&
              p.state.cop0.registers[14] == kernel + 16 &&
              p.state.registers[4] == 0,
          "user privilege cannot reuse a warmed kernel executable translation");
  require((*privileged)->run_slice(p.state, 1, 1).has_value() &&
              p.state.pc.value == kernel + 16 &&
              !(p.state.cop0.registers[12] & 2),
          "ERET restores user mode");
  const let again = (*privileged)->run_slice(p.state, 1, 1);
  require(again && again->reason == jit::stop_reason::guest_exception &&
              p.state.registers[4] == 0,
          "ERET invalidates kernel translations before returning to user mode");
}

fn irq_and_asid() -> void {
  fixture f;
  f.word(0x180, 0x8c620000);
  f.word(0x2000, 0x12345678);
  require(f.state.cop0.translations.write(0, {0, 0x100000000ULL, {0x86, 0}})
              .has_value(),
          "map extended user data");
  f.state.cop0.registers[12] = 0x30;
  require(f.memory->load<std::uint32_t>({0x100000000ULL}) == 0x12345678,
          "warm user extended translation");
  f.state.cop0.registers[12] = 0x8031;
  f.state.cop0.registers[13] = 0x8000;
  f.state.pc = {0x4000};
  f.state.registers[3] = 0x100000000ULL;
  f.state.registers[2] = 99;
  let dispatch = jit::memory_dispatcher::create(*f.memory);
  require(dispatch.has_value(), "create IRQ execution");
  const let result = (*dispatch)->run_slice(f.state, 1, 1);
  require(
      result && result->instructions == 1 &&
          f.state.cop0.registers[14] == 0x4000 &&
          (f.state.cop0.registers[12] & 2) &&
          f.state.registers[2] == 0x12345678,
      "kernel IRQ handlers retain UX-controlled access to user address space");
  fixture a;
  a.word(0, 0x8c620000);
  a.word(4, 0x34050008);
  a.word(8, 0x40855000);
  a.word(12, 0x8c640000);
  a.word(0x2000, 0xaaaa);
  a.word(0x3000, 0xbbbb);
  require(
      a.state.cop0.translations.write(0, {0, 0x4007, {0x86, 0}}).has_value() &&
          a.state.cop0.translations.write(1, {0, 0x4008, {0xc6, 0}})
              .has_value(),
      "install different ASID backings");
  a.state.cop0.registers[10] = 7;
  a.state.pc = {kernel};
  a.state.registers[3] = 0x4000;
  let switched = jit::memory_dispatcher::create(*a.memory, 32);
  require(switched.has_value(), "create ASID execution");
  require((*switched)->run_slice(a.state, 3, 1).has_value() &&
              a.state.registers[2] == 0xaaaa,
          "initial ASID reads first physical page");
  require(
      (*switched)->run_slice(a.state, 1, 1).has_value() &&
          a.state.registers[4] == 0xbbbb,
      "guest EntryHi writes cannot reuse a previous ASID's data cache entry");
}

} // namespace

fn main() -> int {
  data_faults();
  data_delay_slots();
  fetch_slots();
  refill_and_privilege();
  irq_and_asid();
  std::println(
      "CPU faults: cold/native/trace and delay-fetch exceptions passed");
}
