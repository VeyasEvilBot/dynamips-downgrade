#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <memory_resource>
#include <print>
#include <span>
#include <string_view>

#include <dynamips/macros.h>

import jit.interpreter;
import jit.amd64;
import jit.ir;
import jit.memory;
import jit.mips64;

namespace {
namespace cpu = jit::mips64;
namespace interp = jit::interpreter;
namespace mem = jit::memory;
constexpr let rwx =
    mem::permission::read | mem::permission::write | mem::permission::execute;

fn require(bool condition, std::string_view message) -> void {
  if (!condition) {
    std::println(stderr, "interpreter test: {}", message);
    std::exit(1);
  }
}

fn differential() -> void {
  std::array<std::byte, mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 0);
  let compiler = jit::amd64::compiler::create();
  require(memory && compiler, "create differential interpreter/JIT execution");
  std::uint64_t seed = 0x7da3b5421c8876ef;
  const let random = [&seed]() {
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    return seed;
  };
  for (unsigned trial = 0; trial < 128; trial++) {
    std::array<cpu::instruction_word, 34> words{};
    cpu::state interpreted{};
    for (let &reg : interpreted.registers)
      reg = random();
    let compiled = interpreted;
    for (std::size_t i = 0; i < 32; i++) {
      const let rs = static_cast<std::uint32_t>(random() & 31);
      const let rt = static_cast<std::uint32_t>(random() & 31);
      const let rd = static_cast<std::uint32_t>(random() & 31);
      const let immediate = static_cast<std::uint32_t>(random() & 0xffff);
      const let shift = static_cast<std::uint32_t>(random() & 31);
      switch (random() % 18) {
      case 0:
        words[i].bits = (9u << 26) | (rs << 21) | (rt << 16) | immediate;
        break;
      case 1:
        words[i].bits = (0x19u << 26) | (rs << 21) | (rt << 16) | immediate;
        break;
      case 2:
        words[i].bits = (0x0au << 26) | (rs << 21) | (rt << 16) | immediate;
        break;
      case 3:
        words[i].bits = (0x0bu << 26) | (rs << 21) | (rt << 16) | immediate;
        break;
      case 4:
        words[i].bits = (0x0cu << 26) | (rs << 21) | (rt << 16) | immediate;
        break;
      case 5:
        words[i].bits = (0x0du << 26) | (rs << 21) | (rt << 16) | immediate;
        break;
      case 6:
        words[i].bits = (0x0eu << 26) | (rs << 21) | (rt << 16) | immediate;
        break;
      case 7:
        words[i].bits = (rt << 16) | (rd << 11) | (shift << 6);
        break;
      case 8:
        words[i].bits = (rt << 16) | (rd << 11) | (shift << 6) | 2;
        break;
      case 9:
        words[i].bits = (rt << 16) | (rd << 11) | (shift << 6) | 3;
        break;
      case 10:
        words[i].bits = (rs << 21) | (rt << 16) | (rd << 11) | 0x21;
        break;
      case 11:
        words[i].bits = (rs << 21) | (rt << 16) | (rd << 11) | 0x23;
        break;
      case 12:
        words[i].bits = (rs << 21) | (rt << 16) | (rd << 11) | 0x26;
        break;
      case 13:
        words[i].bits = (rs << 21) | (rt << 16) | (rd << 11) | 0x27;
        break;
      case 14:
        words[i].bits = (rs << 21) | (rt << 16) | (rd << 11) | 0x2d;
        break;
      case 15:
        words[i].bits = (rs << 21) | (rt << 16) | (rd << 11) | 0x2b;
        break;
      case 16:
        words[i].bits = (rt << 16) | (rd << 11) | (shift << 6) | 0x3c;
        break;
      case 17:
        words[i].bits = (rs << 21) | (rt << 16) | (rd << 11) | 4;
        break;
      }
    }
    words[32].bits = ((trial & 1) ? 0x50220003u : 0x1422fff0u);
    words[33].bits = 0x24630001;
    alignas(std::max_align_t) std::array<std::byte, 64 * 1024> storage;
    std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                              std::pmr::null_memory_resource()};
    let ir = cpu::translate(words, {0x8000}, arena);
    require(ir && jit::ir::optimize(*ir, jit::ir::fold_constants,
                                    jit::ir::simplify_identities),
            "translate and optimize independently interpreted input");
    let native = compiler->compile(*ir, arena);
    require(native.has_value(), "compile differential input");
    const let reference =
        interp::execute(words, {0x8000}, interpreted, **memory);
    const let actual =
        native->execute(compiled.registers, **memory, &compiled.cop0,
                        &compiled.hilo, &compiled.cop1);
    require(reference && actual && reference->next_pc == actual->next_pc &&
                reference->instructions == actual->instructions &&
                interpreted.registers == compiled.registers,
            "independent interpreter and optimized native JIT agree on "
            "randomized sequences");
  }
}

fn arithmetic_and_rejection() -> void {
  std::array<std::byte, mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 1);
  require(memory.has_value(), "create arithmetic memory");
  cpu::state state{};
  state.registers[1] = 0x7fffffff;
  state.registers[2] = UINT64_MAX;
  constexpr std::array<cpu::instruction_word, 8> words{{{0x24210001},
                                                        {0x64420001},
                                                        {0x00011fc3},
                                                        {0x2c44ffff},
                                                        {0x3c05ffff},
                                                        {0x0005303e},
                                                        {0x00a63826},
                                                        {0x34000009}}};
  const let executed = interp::execute(words, {0x8000}, state, **memory);
  require(executed && executed->instructions == words.size() &&
              executed->next_pc == 0x8020 &&
              executed->accounted_instructions == 0 &&
              state.registers[0] == 0 &&
              state.registers[1] == 0xffffffff80000000ULL &&
              state.registers[2] == 0 && state.registers[3] == UINT64_MAX &&
              state.registers[4] == 1 &&
              state.registers[5] == 0xffffffffffff0000ULL &&
              state.registers[6] == 0xffffffff &&
              state.registers[7] == 0xffffffff0000ffffULL &&
              state.cop0.registers[9] == 0,
          "word sign extension, wrapping arithmetic, signed shifts and r0 "
          "semantics");
  constexpr std::array<cpu::instruction_word, 2> unsupported{
      {{0x24210001}, {0x40816000}}};
  const let before = state.registers;
  const let rejected = interp::execute(unsupported, {0x8000}, state, **memory);
  require(
      !interp::supports(unsupported) && !rejected &&
          rejected.error().code == interp::error_code::unsupported &&
          state.registers == before,
      "unsupported blocks are rejected before any instruction changes state");
  constexpr std::array<cpu::instruction_word, 1> missing_slot{{{0x1000ffff}}};
  constexpr std::array<cpu::instruction_word, 2> nested_branch{
      {{0x1000ffff}, {0x1000ffff}}};
  require(!interp::supports(missing_slot) && !interp::supports(nested_branch),
          "incomplete and nested delay slots require another execution path");
}

fn branch_boundaries() -> void {
  std::array<std::byte, mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 0);
  require(memory.has_value(), "create branch memory");
  for (const bool equal : {false, true}) {
    for (const bool likely : {false, true}) {
      const std::array<cpu::instruction_word, 2> words{
          {{(likely ? 0x50220003u : 0x10220003u)}, {0x24210001}}};
      cpu::state state{};
      state.registers[1] = 4;
      state.registers[2] = equal ? 4 : 5;
      const let executed = interp::execute(words, {0x8000}, state, **memory);
      require(executed && executed->next_pc == (equal ? 0x8010u : 0x8008u) &&
                  executed->instructions == (!equal && likely ? 1u : 2u) &&
                  state.registers[1] == (!equal && likely ? 4u : 5u),
              "branch predicates precede the slot and likely branches annul "
              "untaken slots");
    }
  }
  constexpr std::array<cpu::instruction_word, 2> jalr{
      {{0x00600809}, {0x24220001}}};
  cpu::state state{};
  state.registers[3] = 0x9876543210;
  const let executed = interp::execute(jalr, {0x8000}, state, **memory);
  require(executed && executed->next_pc == 0x9876543210ULL &&
              state.registers[1] == 0x8008 && state.registers[2] == 0x8009,
          "JALR snapshots its target and exposes the link register to the "
          "delay slot");
}

fn signed_likely_branches() -> void {
  std::array<std::byte, mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 0);
  let compiler = jit::amd64::compiler::create();
  require(memory && compiler, "create signed likely branch execution");
  for (const std::uint32_t op : {0x16u, 0x17u}) {
    for (const std::uint64_t operand :
         {0ULL, 1ULL, 0xffffffffffffffffULL, 0x80000000ULL,
          0xffffffff00000000ULL, 0x8000000000000000ULL,
          0x7fffffffffffffffULL}) {
      const bool positive = operand != 0 && (operand >> 63) == 0;
      const bool taken = op == 0x17 ? positive : !positive;
      for (const bool faulting_slot : {false, true}) {
        for (const bool optimized : {false, true}) {
          const std::array<cpu::instruction_word, 3> words{
              {{0x24e70001},
               {(op << 26) | (1u << 21) | 3u},
               {faulting_slot ? 0x8c620000u : 0x24420001u}}};
          require(cpu::is_branch(words[1]) && cpu::has_delay_slot(words[1]) &&
                      interp::supports(words) &&
                      !interp::supports(std::span{words}.first(2)),
                  "signed likely branches require a complete delay slot");
          cpu::state reference{};
          reference.registers[1] = operand;
          reference.registers[2] = 99;
          reference.registers[3] = 0x9000;
          let actual = reference;
          alignas(std::max_align_t) std::array<std::byte, 64 * 1024> storage;
          std::pmr::monotonic_buffer_resource arena{
              storage.data(), storage.size(), std::pmr::null_memory_resource()};
          let ir = cpu::translate(words, {0x8000}, arena);
          require(ir.has_value(), "translate signed likely branch");
          if (optimized)
            require(jit::ir::optimize(*ir, jit::ir::fold_constants,
                                      jit::ir::simplify_identities)
                        .has_value(),
                    "optimize signed likely branch");
          let native = compiler->compile(*ir, arena);
          require(native.has_value(), "compile signed likely branch");
          const let interpreted =
              interp::execute(words, {0x8000}, reference, **memory);
          const let executed =
              native->execute(actual.registers, **memory, &actual.cop0,
                              &actual.hilo, &actual.cop1);
          if (taken && faulting_slot) {
            require(
                !interpreted && !executed &&
                    interpreted.error().code == interp::error_code::memory &&
                    interpreted.error().pc == 0x8008 &&
                    interpreted.error().branch_pc == 0x8004 &&
                    interpreted.error().instructions == 2 &&
                    executed.error().code ==
                        jit::amd64::error_code::memory_fault &&
                    executed.error().pc == 0x8008 &&
                    executed.error().branch_pc == 0x8004 &&
                    executed.error().instructions == 2 &&
                    actual.registers[2] == 99 && actual.registers[7] == 1 &&
                    reference.registers == actual.registers,
                "taken signed likely slots preserve the fault PC, branch PC "
                "and exact completed prefix in both tiers");
          } else {
            const let slots = taken ? 3u : 2u;
            const let next = taken ? 0x8014u : 0x800cu;
            require(interpreted && executed && interpreted->next_pc == next &&
                        executed->next_pc == next &&
                        interpreted->instructions == slots &&
                        executed->instructions == slots &&
                        actual.registers[2] == (taken ? 100u : 99u) &&
                        actual.registers[7] == 1 &&
                        reference.registers == actual.registers,
                    "signed 64-bit likely predicates and annulled slots agree "
                    "in interpreter, optimized and unoptimized native code");
          }
        }
      }
    }
    alignas(std::max_align_t) std::array<std::byte, 64 * 1024> storage;
    std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                              std::pmr::null_memory_resource()};
    const std::array<cpu::instruction_word, 2> invalid{
        {{(op << 26) | (1u << 21) | (1u << 16) | 3u}, {0}}};
    const let translated = cpu::translate(invalid, {0x8000}, arena);
    require(!interp::supports(invalid) && !translated &&
                translated.error().code == cpu::error_code::invalid_encoding,
            "BLEZL/BGTZL reject nonzero reserved rt fields");
  }
}

fn memory_faults_and_aliases() -> void {
  std::array<std::byte, mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 2);
  require(memory && (*memory)->map_ram({0x1000}, {0}, rwx).has_value() &&
              (*memory)->map_ram({0x3000}, {0}, rwx).has_value(),
          "map interpreter RAM aliases");
  constexpr std::array<cpu::instruction_word, 4> words{
      {{0x24210001}, {0xac610000}, {0x14220001}, {0x8c850000}}};
  cpu::state state{};
  state.registers[3] = 0x3100;
  state.registers[4] = 0x9000;
  const let fault = interp::execute(words, {0x8000}, state, **memory);
  require(!fault && fault.error().code == interp::error_code::memory &&
              fault.error().pc == 0x800c && fault.error().branch_pc == 0x8008 &&
              fault.error().completed_instructions == 3 &&
              fault.error().access.address.value == 0x9000 &&
              state.registers[1] == 1 &&
              (*memory)->load<std::uint32_t>({0x1100}) == 1,
          "delay-slot faults preserve earlier stores and report the completed "
          "prefix");
  constexpr std::array<cpu::instruction_word, 2> changing{
      {{0xac620000}, {0x24040001}}};
  const std::array code{std::span<const std::byte>{ram}.first(8)};
  state = {};
  state.registers[3] = 0x3004;
  state.registers[2] = 0x24040009;
  const let changed =
      interp::execute(changing, {0x1000}, state, **memory, code);
  require(changed && changed->instructions == 1 && changed->next_pc == 0x1004 &&
              state.registers[4] == 0 &&
              (*memory)->fetch<std::uint32_t>({0x1004}) == 0x24040009,
          "writes through executable aliases stop before stale decoded "
          "instructions");

  constexpr std::array<cpu::instruction_word, 3> linked{
      {{0xc0610000}, {0x24210001}, {0xe0610000}}};
  state = {};
  state.registers[3] = 0x3100;
  const let linked_result = interp::execute(linked, {0x8000}, state, **memory);
  require(linked_result && state.registers[1] == 1 &&
              (*memory)->load<std::uint32_t>({0x1100}) == 2,
          "LL/SC operates on physical reservations and reports success in the "
          "source register");
}

fn merged_accesses() -> void {
  std::array<std::byte, 2 * mem::page_size> ram{};
  for (std::size_t i = 0; i < ram.size(); i++)
    ram[i] = static_cast<std::byte>(i & 255);
  let memory = mem::address_space::create(ram, 2);
  require(memory && (*memory)->map_ram({0x1000}, {0}, rwx).has_value(),
          "map merged interpreter accesses");
  for (const bool word : {false, true}) {
    const std::array<cpu::instruction_word, 2> words{
        {{word ? 0x88220000u : 0x68220000u},
         {word ? 0x98220003u : 0x6c220007u}}};
    cpu::state state{};
    state.registers[1] = 0x1ffd;
    const let fault = interp::execute(words, {0x8000}, state, **memory);
    require(
        !fault && fault.error().pc == 0x8004 &&
            fault.error().completed_instructions == 1 &&
            fault.error().access.address.value == (word ? 0x2000u : 0x2004u) &&
            state.registers[2] ==
                (word ? 0xfffffffffdfeff00ULL : 0xfdfeff0000000000ULL),
        "faulting paired merge retains the first instruction's partial result");
  }
  require((*memory)->map_ram({0x2000}, {mem::page_size}, rwx).has_value(),
          "map second merged source page");
  constexpr std::array<cpu::instruction_word, 2> pair{
      {{0x68220000}, {0x6c220007}}};
  cpu::state state{};
  state.registers[1] = 0x1ffd;
  const let joined = interp::execute(pair, {0x8000}, state, **memory);
  require(joined && state.registers[2] == 0xfdfeff0001020304ULL,
          "paired doubleword merge spans distinct physical source pages");
}

fn mmio_clock() -> void {
  struct observation {
    cpu::state *state;
    std::uint64_t count = UINT64_MAX;
  };

  std::array<std::byte, mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 1);
  require(memory.has_value(), "create MMIO interpreter memory");
  cpu::state state{};
  observation observed{&state};
  mem::device device{
      &observed,
      [](void *opaque, mem::device_access) noexcept
          -> std::expected<std::uint64_t, mem::device_error> {
        let &value = *static_cast<observation *>(opaque);
        value.count = value.state->cop0.registers[9];
        return 7;
      },
      [](void *, mem::device_access,
         std::uint64_t) noexcept -> std::expected<void, mem::device_error> {
        return std::unexpected(mem::device_error::unsupported_size);
      }};
  require((*memory)
              ->map_device({0x9000}, {0x100000}, device, mem::permission::read)
              .has_value(),
          "map observed MMIO read");
  constexpr std::array<cpu::instruction_word, 4> words{
      {{0x24210001}, {0x24210001}, {0x8c820000}, {0x24030009}}};
  state.registers[4] = 0x9000;
  const let executed = interp::execute(words, {0x8000}, state, **memory);
  require(
      executed && executed->instructions == 3 &&
          executed->accounted_instructions == 2 &&
          executed->next_pc == 0x800c && observed.count == 1 &&
          state.registers[2] == 7 && state.registers[3] == 0,
      "MMIO sees completed virtual time and exits before the next instruction");
}
} // namespace

fn main() -> int {
  differential();
  arithmetic_and_rejection();
  branch_boundaries();
  signed_likely_branches();
  memory_faults_and_aliases();
  merged_accesses();
  mmio_clock();
  std::println("interpreter: arithmetic, branches, memory, aliases and MMIO "
               "clock checks passed");
}
