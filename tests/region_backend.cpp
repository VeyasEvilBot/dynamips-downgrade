#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <memory_resource>
#include <print>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>
#include <dynamips/macros.h>
import jit.amd64;
import jit.ir;
import jit.memory;
import jit.mips64;

namespace {
namespace cpu = jit::mips64;
namespace mem = jit::memory;
namespace amd = jit::amd64;
constexpr let rwx =
    mem::permission::read | mem::permission::write | mem::permission::execute;
fn require(bool good, std::string_view message) -> void {
  if (!good) {
    std::println(stderr, "region backend test: {}", message);
    std::exit(1);
  }
}

constexpr fn jal(std::uint64_t target) -> std::uint32_t {
  return 0x0c000000u | static_cast<std::uint32_t>((target >> 2) & 0x3ffffff);
}

struct fixture {
  std::vector<std::byte> ram = std::vector<std::byte>(64 * mem::page_size);
  std::vector<std::byte> scratch = std::vector<std::byte>(1024 * 1024);
  std::pmr::monotonic_buffer_resource arena{scratch.data(), scratch.size(),
                                            std::pmr::null_memory_resource()};
  std::pmr::vector<jit::ir::block> regions{&arena};
  std::vector<amd::compiled_block> reference;
  std::array<std::span<const std::byte>, 64> ranges{};
  std::unique_ptr<mem::address_space> memory;
  const amd::compiler &compiler;

  explicit fixture(const amd::compiler &backend) : compiler(backend) {
    let space = mem::address_space::create(ram, 4);
    require(space.has_value(), "create bounded region memory");
    memory = std::move(*space);
    regions.reserve(64);
    reference.reserve(64);
  }

  template <std::size_t N>
  fn add(std::uint64_t pc, const std::array<std::uint32_t, N> &bits) -> void {
    std::array<cpu::instruction_word, N> words;
    for (std::size_t i = 0; i < N; i++)
      words[i] = {bits[i]};
    let block = cpu::translate(words, {pc}, arena);
    require(block.has_value(), "translate region block");
    require(jit::ir::optimize(*block, jit::ir::fold_constants,
                              jit::ir::simplify_identities)
                .has_value(),
            "optimize region block");
    const let index = regions.size();
    require(index < ranges.size(), "bounded test regions");
    for (std::size_t i = 0; i < block->info().instruction_count; i++)
      for (unsigned byte = 0; byte < 4; byte++)
        ram[index * mem::page_size + i * 4 + byte] =
            static_cast<std::byte>(bits[i] >> ((3 - byte) * 8));
    ranges[index] = std::span<const std::byte>{ram}.subspan(
        index * mem::page_size, block->info().instruction_count * 4);
    let native = compiler.compile(*block, arena);
    require(native.has_value(), "compile region block reference");
    reference.push_back(std::move(*native));
    regions.push_back(std::move(*block));
  }
  fn compile(bool path_only = false,
               std::optional<std::size_t> backedge = std::nullopt)
      -> amd::compiled_block {
    let result =
        compiler.compile_trace(regions, arena, true, true, path_only, backedge);
    require(result.has_value(), "compile noncontiguous native region");
    return std::move(*result);
  }
  fn code() const noexcept -> std::span<const std::span<const std::byte>> {
    return std::span{ranges}.first(regions.size());
  }
};

struct progress {
  std::uint64_t pc;
  std::uint32_t instructions = 0, blocks = 0;
};

fn run_reference(fixture &f, cpu::state &state, std::uint64_t pc,
                   std::uint64_t budget, std::uint32_t limit,
                   std::uint64_t deadline) -> progress {
  progress result{pc};
  while (result.blocks < limit) {
    std::size_t index = f.regions.size();
    for (std::size_t i = 0; i < f.regions.size(); i++)
      if (f.regions[i].info().start_pc == result.pc)
        index = i;
    if (index == f.regions.size() || f.regions[index].info().instruction_count >
                                         budget - result.instructions)
      break;
    const let executed = f.reference[index].execute(
        state.registers, *f.memory, &state.cop0, &state.hilo, &state.cop1);
    require(executed.has_value(), "execute nested block reference");
    state.cop0.advance_cycles(executed->instructions -
                              executed->accounted_instructions);
    result.pc = executed->next_pc;
    result.instructions += executed->instructions;
    result.blocks += executed->blocks;
    if (result.instructions >= deadline)
      break;
  }
  return result;
}
fn nested_calls(const amd::compiler &compiler, std::uint64_t base) -> void {
  fixture f{compiler};
  f.add(base + 0x1000,
        std::array{0x24210001u, jal(base + 0x3000), 0x24420001u});
  f.add(base + 0x3000,
        std::array{0x03e0802du, jal(base + 0x5000), 0x24630001u});
  f.add(base + 0x100c, std::array{0x24a50001u, 0x02800008u, 0x24c60001u});
  f.add(base + 0x300c, std::array{0x0200f82du, 0x03e00008u, 0x24840001u});
  // Same low32 bits as the real leaf for the high-address test. It must not
  // match.
  if (base == 0x180000000ULL)
    f.add(0xffffffff80005000ULL, std::array{0x03e00008u, 0x24070063u});
  f.add(base + 0x5000, std::array{0x03e00008u, 0x24e70001u});
  let native = f.compile();
  for (const std::uint64_t budget :
       {0ULL, 1ULL, 2ULL, 3ULL, 5ULL, 8ULL, 14ULL, 64ULL}) {
    for (const std::uint32_t blocks : {1u, 2u, 3u, 5u, 64u}) {
      for (const std::uint64_t deadline :
           {1ULL, 3ULL, 4ULL, 7ULL, 14ULL, 0xffffffffffffffffULL}) {
        cpu::state actual{};
        actual.registers[20] = base + 0x9000;
        actual.cop0.registers[11] = static_cast<std::uint32_t>(deadline);
        let expected = actual;
        const let result = native.execute(
            actual.registers, *f.memory, &actual.cop0, &actual.hilo,
            &actual.cop1, f.code(), budget, blocks, deadline);
        require(result.has_value(), "execute bounded nested calls");
        actual.cop0.advance_cycles(result->instructions -
                                   result->accounted_instructions);
        const let reference =
            run_reference(f, expected, base + 0x1000, budget, blocks, deadline);
        require(result->next_pc == reference.pc &&
                    result->instructions == reference.instructions &&
                    result->blocks == reference.blocks &&
                    actual.registers == expected.registers &&
                    actual.cop0.registers == expected.cop0.registers &&
                    actual.cop0.half_cycle == expected.cop0.half_cycle,
                "nested calls/returns preserve prefixes, budgets, deadlines "
                "and full64 PCs");
        if (budget == 64 && blocks == 64 && deadline == UINT64_MAX)
          require(result->instructions == 14 && result->blocks == 5 &&
                      result->next_pc == base + 0x9000 &&
                      actual.registers[7] == 1,
                  "internal nested returns execute every region before "
                  "unmatched exit");
      }
    }
  }
}
fn indirect_snapshots(const amd::compiler &compiler, bool path_only = false)
    -> void {
  constexpr std::uint64_t base = 0x180000000ULL;
  fixture f{compiler};
  f.add(base + 0x1000, std::array{0x0060f809u, 0x02a0182du});
  f.add(base + 0x3000, std::array{0x03e00008u, 0x02c0f82du});
  f.add(base + 0x1008, std::array{0x24090009u, 0x02800008u, 0u});
  let native = f.compile(path_only);
  for (const std::uint64_t target :
       {base + 0x3000, base + 0x3001, base + 0x7000}) {
    cpu::state state{};
    state.registers[3] = target;
    state.registers[20] = state.registers[21] = base + 0x9000;
    state.registers[22] = base + 0xb000;
    const let result =
        native.execute(state.registers, *f.memory, &state.cop0, &state.hilo,
                       &state.cop1, f.code(), 100, 64);
    require(result && state.registers[3] == base + 0x9000,
            "JALR snapshots its old target before an aliased delay-slot write");
    if (target == base + 0x3000)
      require(result->next_pc == base + 0x9000 && result->instructions == 7 &&
                  result->blocks == 3 && state.registers[31] == base + 0xb000 &&
                  state.registers[9] == 9,
              "internal JR snapshots RA before its delay slot overwrites RA");
    else
      require(
          result->next_pc == target && result->instructions == 2 &&
              result->blocks == 1 && state.registers[31] == base + 0x1008 &&
              state.registers[9] == 0,
          "unknown or unaligned indirect targets return unchanged to dispatch");
  }
}
fn faulting_return(const amd::compiler &compiler, bool path_only = false)
    -> void {
  constexpr std::uint64_t base = 0x180000000ULL;
  fixture f{compiler};
  f.add(base + 0x1000, std::array{jal(base + 0x3000), 0u});
  f.add(base + 0x3000, std::array{jal(base + 0x5000), 0u});
  f.add(base + 0x5000, std::array{0x03e00008u, 0x8c620000u});
  f.add(base + 0x3008, std::array{0x24090009u});
  let native = f.compile(path_only);
  cpu::state state{};
  state.registers[3] = 0xdead0000;
  const let result =
      native.execute(state.registers, *f.memory, &state.cop0, &state.hilo,
                     &state.cop1, f.code(), 100, 64);
  require(!result && result.error().code == amd::error_code::memory_fault &&
              result.error().pc == base + 0x5004 &&
              result.error().branch_pc == base + 0x5000 &&
              result.error().instructions == 5 && result.error().blocks == 3 &&
              state.registers[9] == 0,
          "noncontiguous return-slot faults preserve exact total prefix and "
          "branch PC");
}
fn code_write_guard(const amd::compiler &compiler, bool path_only = false)
    -> void {
  fixture f{compiler};
  f.add(0x1000, std::array{0x02800008u, 0xac620000u});
  f.add(0x3000, std::array{0x24090001u});
  require(f.memory->map_ram({0x7000}, {mem::page_size}, rwx).has_value(),
          "map a writable alias of the internal target's executable backing");
  let native = f.compile(path_only);
  cpu::state state{};
  state.registers[20] = 0x3000;
  state.registers[2] = 0x24090063;
  state.registers[3] = 0x7000;
  const let result =
      native.execute(state.registers, *f.memory, &state.cop0, &state.hilo,
                     &state.cop1, f.code(), 100, 64);
  require(
      result && result->next_pc == 0x3000 && result->instructions == 2 &&
          result->blocks == 1 && state.registers[9] == 0 &&
          f.memory->load<std::uint32_t>({0x7000}) == 0x24090063,
      "a return-slot code write exits before chaining into the stale target");
}
fn cpu_barrier(const amd::compiler &compiler, bool path_only = false)
    -> void {
  fixture f{compiler};
  f.add(0x1000, std::array{0x00600008u, 0u});
  f.add(0x3000, std::array{0x40044800u});
  let native = f.compile(path_only);
  cpu::state state{};
  state.registers[3] = 0x3000;
  state.registers[4] = 99;
  const let result =
      native.execute(state.registers, *f.memory, &state.cop0, &state.hilo,
                     &state.cop1, f.code(), 100, 64);
  require(result && result->next_pc == 0x3000 && result->instructions == 2 &&
              result->blocks == 1 && state.registers[4] == 99,
          "indirect chaining preserves the existing CP0 block-entry barrier");
}
fn selected_path(const amd::compiler &compiler, bool likely = false) -> void {
  fixture f{compiler};
  f.add(0x1000, std::array{likely ? 0x502007ffu : 0x102007ffu, 0x24420001u});
  f.add(0x3000, std::array{0x00600008u, 0x24840001u});
  f.add(0x5000, std::array{0x08000400u, 0x24a50001u});
  let native = f.compile(true);
  for (const std::uint64_t direction : {0ULL, 1ULL})
    for (const std::uint64_t target : {0x5000ULL, 0x7000ULL, 0x100005000ULL})
      for (const std::uint64_t budget : {0ULL, 1ULL, 2ULL, 3ULL, 5ULL, 6ULL,
                                         7ULL, 8ULL, 12ULL, 13ULL, 64ULL})
        for (const std::uint32_t blocks : {1u, 2u, 3u, 4u, 6u, 7u, 64u})
          for (const std::uint64_t deadline :
               {0ULL, 1ULL, 3ULL, 6ULL, 7ULL, 12ULL, 13ULL,
                0xffffffffffffffffULL}) {
            cpu::state actual{};
            actual.registers[1] = direction;
            actual.registers[3] = target;
            let expected = actual;
            const let result = native.execute(
                actual.registers, *f.memory, &actual.cop0, &actual.hilo,
                &actual.cop1, f.code(), budget, blocks, deadline);
            require(result.has_value(), "execute a selected hot path");
            actual.cop0.advance_cycles(result->instructions -
                                       result->accounted_instructions);
            const let reference =
                run_reference(f, expected, 0x1000, budget, blocks, deadline);
            require(result->next_pc == reference.pc &&
                        result->instructions == reference.instructions &&
                        result->blocks == reference.blocks &&
                        actual.registers == expected.registers &&
                        actual.cop0.registers == expected.cop0.registers &&
                        actual.cop0.half_cycle == expected.cop0.half_cycle,
                    "hot path preserves branch alternatives, unexpected full64 "
                    "JR targets, limits and cycle deadlines");
          }
  cpu::state alternate{};
  alternate.registers[3] = 0x1000;
  const let exited =
      native.execute(alternate.registers, *f.memory, &alternate.cop0,
                     &alternate.hilo, &alternate.cop1, f.code(), 64, 64);
  require(
      exited && exited->next_pc == 0x1000 && exited->blocks == 2 &&
          exited->instructions == 4 && alternate.registers[5] == 0,
      "JR to an included non-successor block exits instead of graph chaining");
}

constexpr fn jump_to(std::uint64_t target) -> std::uint32_t {
  return 0x08000000u | static_cast<std::uint32_t>((target >> 2) & 0x3ffffff);
}
fn contextual_calls(const amd::compiler &compiler, std::uint64_t base)
    -> void {
  fixture f{compiler};
  f.add(base + 0x1000, std::array{jal(base + 0x3000), 0x24210001u});
  f.add(base + 0x3000, std::array{0x03e00008u, 0x24420001u});
  f.add(base + 0x1008, std::array{jal(base + 0x3000), 0x24630001u});
  f.add(base + 0x3000, std::array{0x03e00008u, 0x24420001u});
  f.add(base + 0x1010, std::array{jump_to(base + 0x1000), 0x24840001u});
  f.ranges[3] = f.ranges[1];
  let native = f.compile(true);
  for (const std::uint64_t budget :
       {0ULL, 1ULL, 2ULL, 3ULL, 5ULL, 7ULL, 9ULL, 10ULL, 11ULL, 20ULL, 128ULL})
    for (const std::uint32_t limit : {1u, 2u, 3u, 4u, 5u, 6u, 10u, 64u})
      for (const std::uint64_t deadline :
           {0ULL, 1ULL, 5ULL, 10ULL, 11ULL, 20ULL, 0xffffffffffffffffULL}) {
        cpu::state actual{};
        let expected = actual;
        const let result = native.execute(
            actual.registers, *f.memory, &actual.cop0, &actual.hilo,
            &actual.cop1, f.code(), budget, limit, deadline);
        require(result.has_value(), "execute contextual duplicate helpers");
        actual.cop0.advance_cycles(result->instructions -
                                   result->accounted_instructions);
        const let reference =
            run_reference(f, expected, base + 0x1000, budget, limit, deadline);
        require(result->next_pc == reference.pc &&
                    result->instructions == reference.instructions &&
                    result->blocks == reference.blocks &&
                    actual.registers == expected.registers &&
                    actual.cop0.registers == expected.cop0.registers &&
                    actual.cop0.half_cycle == expected.cop0.half_cycle,
                "duplicate helper instances keep contextual returns, loops, "
                "limits and exact counts");
      }
}
fn alternate_context_return(const amd::compiler &compiler) -> void {
  constexpr std::uint64_t base = 0x180000000ULL;
  fixture f{compiler};
  f.add(base + 0x1000, std::array{jal(base + 0x3000), 0x24210001u});
  f.add(base + 0x3000, std::array{0x02800008u, 0x24420001u});
  f.add(base + 0x1008, std::array{jal(base + 0x3000), 0x02a0a02du});
  f.add(base + 0x3000, std::array{0x02800008u, 0x24420001u});
  f.add(base + 0x1010, std::array{jump_to(base + 0x1000), 0x24840001u});
  f.ranges[3] = f.ranges[1];
  let native = f.compile(true);
  for (const std::uint64_t first :
       {base + 0x1008, base + 0x1010, std::uint64_t{0xffffffff80001008ULL}}) {
    cpu::state state{};
    state.registers[20] = first;
    state.registers[21] = base + 0x7000;
    const let result =
        native.execute(state.registers, *f.memory, &state.cop0, &state.hilo,
                       &state.cop1, f.code(), 100, 64);
    const bool second = first == base + 0x1008;
    require(result && result->next_pc == (second ? base + 0x7000 : first) &&
                result->instructions == (second ? 8u : 4u) &&
                result->blocks == (second ? 4u : 2u) &&
                state.registers[2] == (second ? 2u : 1u) &&
                state.registers[4] == 0,
            "an unexpected contextual/full64 return exits even when another "
            "label shares its PC");
  }
}
fn second_instance_fault(const amd::compiler &compiler, bool delay) -> void {
  fixture f{compiler};
  f.add(0x1000, std::array{jal(0x3000), 0u});
  if (delay)
    f.add(0x3000, std::array{0x03e00008u, 0x8cc50000u});
  else
    f.add(0x3000, std::array{0x8cc50000u, 0x03e00008u, 0u});
  f.add(0x1008, std::array{jal(0x3000), 0x3c06deadu});
  if (delay)
    f.add(0x3000, std::array{0x03e00008u, 0x8cc50000u});
  else
    f.add(0x3000, std::array{0x8cc50000u, 0x03e00008u, 0u});
  f.add(0x1010, std::array{jump_to(0x1000), 0x24840001u});
  f.ranges[3] = f.ranges[1];
  require(f.memory->map_ram({0x7000}, {20 * mem::page_size}, rwx).has_value() &&
              f.memory->store<std::uint32_t>({0x7000}, 123).has_value(),
          "install the first helper's valid load");
  let native = f.compile(true);
  cpu::state state{};
  state.registers[6] = 0x7000;
  const let result =
      native.execute(state.registers, *f.memory, &state.cop0, &state.hilo,
                     &state.cop1, f.code(), 100, 64);
  require(
      !result && result.error().code == amd::error_code::memory_fault &&
          result.error().pc == (delay ? 0x3004u : 0x3000u) &&
          result.error().instructions == 7 && result.error().blocks == 4 &&
          result.error().branch_pc ==
              (delay ? std::optional<std::uint64_t>{0x3000} : std::nullopt) &&
          state.registers[5] == 123 && state.registers[4] == 0,
      "a second helper instance reports the same fault PC with its longer "
      "execution prefix");
}
fn second_instance_code_write(const amd::compiler &compiler) -> void {
  fixture f{compiler};
  f.add(0x1000, std::array{jal(0x3000), 0u});
  f.add(0x3000, std::array{0xacc50000u, 0x03e00008u, 0x24420001u});
  f.add(0x1008, std::array{jal(0x3000), 0x34069000u});
  f.add(0x3000, std::array{0xacc50000u, 0x03e00008u, 0x24420001u});
  f.add(0x1010, std::array{jump_to(0x1000), 0x24840001u});
  f.ranges[3] = f.ranges[1];
  require(f.memory->map_ram({0x7000}, {20 * mem::page_size}, rwx).has_value() &&
              f.memory->map_ram({0x9000}, {mem::page_size}, rwx).has_value(),
          "map ordinary data and a shared helper-code alias");
  let native = f.compile(true);
  cpu::state state{};
  state.registers[5] = 0;
  state.registers[6] = 0x7000;
  const let result =
      native.execute(state.registers, *f.memory, &state.cop0, &state.hilo,
                     &state.cop1, f.code(), 100, 64);
  require(result && result->next_pc == 0x3004 && result->instructions == 8 &&
              result->blocks == 4 && state.registers[2] == 1 &&
              state.registers[4] == 0 &&
              f.memory->load<std::uint32_t>({0x9000}) == 0,
          "a code write in a duplicate helper exits before its return and "
          "later context");
}
fn overlapping_context(const amd::compiler &compiler) -> void {
  fixture f{compiler};
  f.add(0x1000, std::array{0u, jump_to(0x1004), 0u});
  f.add(0x1004, std::array{jump_to(0x1004), 0u});
  f.ranges[1] = f.ranges[0].subspan(4);
  require(!compiler.compile_trace(f.regions, f.arena),
          "a general graph still rejects overlapping instruction ranges");
  let native = f.compile(true);
  cpu::state state{};
  const let result =
      native.execute(state.registers, *f.memory, &state.cop0, &state.hilo,
                     &state.cop1, f.code(), 100, 64);
  require(result && result->next_pc == 0x1004 && result->instructions == 5 &&
              result->blocks == 2,
          "ordered overlapping contexts enter only their expected next label");
}
fn nested_cycle(fixture &f, bool loads = false, bool stores = false,
                  bool delay_load = false) -> void {
  const let leaf =
      loads ? (delay_load ? std::array{0x254a0004u, 0x03e00008u, 0x8d440000u}
                          : std::array{0x8d440000u, 0x03e00008u, 0x254a0004u})
            : std::array{0x24630001u, 0x03e00008u, 0u};
  f.add(0x1000, std::array{0x24210001u, jal(0x3000), 0u});
  f.add(0x3000, std::array{0x03e08021u, jal(0x5000), 0u});
  f.add(0x5000, leaf);
  f.add(0x300c, std::array{0x0200f821u, 0x03e00008u, 0u});
  f.add(0x100c, std::array{0x24420001u, jal(0x3000), 0u});
  f.add(0x3000, std::array{0x03e08021u, jal(0x5000), 0u});
  f.add(0x5000, leaf);
  f.add(0x300c, std::array{0x0200f821u, 0x03e00008u, 0u});
  if (stores)
    f.add(0x1018, std::array{0xacc50000u, jump_to(0x100c), 0x24c60004u});
  else
    f.add(0x1018, std::array{jump_to(0x100c), 0u});
}

fn internal_cycle_limits(const amd::compiler &compiler) -> void {
  fixture f{compiler};
  nested_cycle(f);
  let native = f.compile(true, 4);
  for (const bool half : {false, true})
    for (const let count : {0u, 0x7ffffffdu, 0xfffffffdu})
      for (const std::uint64_t budget :
           {0ULL, 1ULL, 2ULL, 3ULL, 11ULL, 12ULL, 13ULL, 25ULL, 26ULL, 27ULL,
            40ULL, 80ULL, 192ULL})
        for (const std::uint32_t blocks :
             {0u, 1u, 3u, 4u, 5u, 8u, 9u, 14u, 63u, 64u})
          for (const std::uint64_t deadline :
               {1ULL, 3ULL, 13ULL, 26ULL, 50ULL, 0xffffffffffffffffULL}) {
            cpu::state actual{};
            actual.cop0.registers[9] = count;
            actual.cop0.registers[11] = static_cast<std::uint32_t>(count + 7);
            actual.cop0.half_cycle = half;
            let expected = actual;
            const let result = native.execute(
                actual.registers, *f.memory, &actual.cop0, &actual.hilo,
                &actual.cop1, f.code(), budget, blocks, deadline);
            require(result.has_value(), "execute internal nested cycle");
            actual.cop0.advance_cycles(result->instructions -
                                       result->accounted_instructions);
            const let reference =
                run_reference(f, expected, 0x1000, budget, blocks, deadline);
            require(result->next_pc == reference.pc &&
                        result->instructions == reference.instructions &&
                        result->blocks == reference.blocks &&
                        actual.registers == expected.registers &&
                        actual.cop0.registers == expected.cop0.registers &&
                        actual.cop0.half_cycle == expected.cop0.half_cycle,
                    "internal backedge preserves scalar prefixes, block "
                    "limits, Count rollover and Compare phases");
          }
  cpu::state state{};
  const let complete =
      native.execute(state.registers, *f.memory, &state.cop0, &state.hilo,
                     &state.cop1, f.code(), 1000, 64);
  require(complete && complete->instructions == 180 && complete->blocks == 64 &&
              complete->next_pc == 0x100c && state.registers[1] == 1 &&
              state.registers[2] == 12 && state.registers[3] == 13,
          "the preheader executes once while a nested internal cycle repeats "
          "to the unchanged 64-block service boundary");
  require(!compiler.compile_trace(f.regions, f.arena, true, true, false, 4) &&
              !compiler.compile_trace(f.regions, f.arena, true, true, true, 9),
          "reject internal indexes outside an ordered path");
}

fn internal_cycle_indirect_guard(const amd::compiler &compiler) -> void {
  fixture f{compiler};
  f.add(0x1000, std::array{0x24210001u, jump_to(0x3000), 0u});
  f.add(0x3000, std::array{0x24420001u, 0x02800008u, 0u});
  let native = f.compile(true, 1);
  for (const let target : {0x3000ULL, 0x1000ULL, 0x100003000ULL}) {
    cpu::state state{};
    state.registers[20] = target;
    const let result =
        native.execute(state.registers, *f.memory, &state.cop0, &state.hilo,
                       &state.cop1, f.code(), 1000, 64);
    require(result && result->next_pc == target && state.registers[1] == 1 &&
                result->blocks == (target == 0x3000 ? 64u : 2u) &&
                state.registers[2] == (target == 0x3000 ? 63u : 1u),
            "the last JR guards the chosen internal label, including root "
            "and high-address target mismatches");
  }
}

fn internal_cycle_fault(const amd::compiler &compiler) -> void {
  for (const bool delay : {false, true}) {
    fixture f{compiler};
    nested_cycle(f, true, false, delay);
    require(f.memory->map_ram({0x6000}, {20 * mem::page_size}, rwx).has_value(),
            "map the two successful loads before an internal-cycle fault");
    require(f.memory->store<std::uint32_t>({0x6ff8}, 0x11).has_value() &&
                f.memory->store<std::uint32_t>({0x6ffc}, 0x22).has_value(),
            "install fault-prefix data");
    let native = f.compile(true, 4);
    cpu::state state{};
    state.registers[10] = delay ? 0x6ff4 : 0x6ff8;
    const let result =
        native.execute(state.registers, *f.memory, &state.cop0, &state.hilo,
                       &state.cop1, f.code(), 1000, 64);
    const let prefix = delay ? 34u : 32u;
    require(!result && result.error().code == amd::error_code::memory_fault &&
                result.error().pc == 0x5000 + (delay ? 8u : 0u) &&
                (delay ? result.error().branch_pc == 0x5004
                       : !result.error().branch_pc) &&
                result.error().instructions == prefix &&
                result.error().completed_instructions == prefix &&
                result.error().blocks == 12 && state.registers[1] == 1 &&
                state.registers[2] == 2 && state.registers[4] == 0x22 &&
                state.registers[10] == 0x7000,
            "later internal-cycle ordinary/delay load faults retain the "
            "total prefix, branch PC and earlier architectural commits");
  }
}

fn internal_cycle_branch_guard(const amd::compiler &compiler) -> void {
  fixture f{compiler};
  f.add(0x1000, std::array{0x24210001u, jump_to(0x3000), 0u});
  f.add(0x3000, std::array{0x2442ffffu, 0x1440fffeu, 0u});
  let native = f.compile(true, 1);
  cpu::state state{};
  state.registers[2] = 3;
  const let result =
      native.execute(state.registers, *f.memory, &state.cop0, &state.hilo,
                     &state.cop1, f.code(), 1000, 64);
  require(result && result->instructions == 12 && result->blocks == 4 &&
              result->next_pc == 0x300c && state.registers[1] == 1 &&
              state.registers[2] == 0,
          "an internal conditional backedge retests the actual branch and "
          "takes its unobserved fallthrough precisely");
}

fn internal_cycle_code_write(const amd::compiler &compiler) -> void {
  fixture f{compiler};
  nested_cycle(f, false, true);
  f.ranges[5] = f.ranges[1];
  require(f.memory->map_ram({0x6000}, {20 * mem::page_size}, rwx).has_value() &&
              f.memory->map_ram({0x7000}, {mem::page_size}, rwx).has_value(),
          "map data followed by an alias of repeated helper code");
  let native = f.compile(true, 4);
  cpu::state state{};
  state.registers[6] = 0x6ffc;
  const let result =
      native.execute(state.registers, *f.memory, &state.cop0, &state.hilo,
                     &state.cop1, f.code(), 1000, 64);
  require(result && result->next_pc == 0x101c && result->instructions == 40 &&
              result->blocks == 14 && state.registers[1] == 1 &&
              state.registers[2] == 2 && state.registers[3] == 3 &&
              state.registers[6] == 0x7000 &&
              f.memory->load<std::uint32_t>({0x7000}) == 0,
          "a write after repeating the internal cycle exits before its "
          "branch and before any stale helper execution");
}

fn raw_trace_requires_context(const amd::compiler &compiler) -> void {
  fixture f{compiler};
  f.add(0x1000, std::array{0x08001400u, 0u});
  f.add(0x5000, std::array{0x24420001u, 0x08002400u, 0u});
  let code = f.compile(true);
  cpu::state state{};
  const let raw = code.run(state.registers);
  require(!raw && raw.error().code == amd::error_code::memory_required,
          "a pure fused trace rejects raw run without an execution context");
  amd::compiled_block::execution_session session{state, *f.memory};
  const let executed = code.execute(session, f.code(), 5, 2, UINT64_MAX,
                                    mem::range_stability::stable);
  require(executed && executed->instructions == 5 && executed->blocks == 2 &&
              executed->next_pc == 0x9000 && state.registers[2] == 1,
          "the same trace executes with a session and explicit stable ranges");
}

fn invalid_regions(const amd::compiler &compiler) -> void {
  for (const bool overlap : {false, true}) {
    fixture f{compiler};
    f.add(0x1000, std::array{0u, 0u});
    f.add(overlap ? 0x1004 : 0x1000, std::array{0u});
    require(!compiler.compile_trace(f.regions, f.arena),
            "ambiguous duplicate/overlapping region addresses are rejected");
  }
  fixture f{compiler};
  std::array<std::uint32_t, 129> words{};
  f.add(0x1000, words);
  f.add(0x3000, words);
  require(!compiler.compile_trace(f.regions, f.arena),
          "regions exceeding the 256-instruction static bound are rejected");
}
} // namespace

fn main() -> int {
  let compiler = amd::compiler::create();
  require(compiler.has_value(), "create guarded-region backend");
  internal_cycle_limits(*compiler);
  internal_cycle_indirect_guard(*compiler);
  internal_cycle_branch_guard(*compiler);
  internal_cycle_fault(*compiler);
  internal_cycle_code_write(*compiler);
  nested_calls(*compiler, 0);
  nested_calls(*compiler, 0x180000000ULL);
  indirect_snapshots(*compiler);
  faulting_return(*compiler);
  code_write_guard(*compiler);
  cpu_barrier(*compiler);
  invalid_regions(*compiler);
  raw_trace_requires_context(*compiler);
  overlapping_context(*compiler);
  contextual_calls(*compiler, 0);
  contextual_calls(*compiler, 0x180000000ULL);
  contextual_calls(*compiler, 0xffffffff80000000ULL);
  alternate_context_return(*compiler);
  second_instance_fault(*compiler, false);
  second_instance_fault(*compiler, true);
  second_instance_code_write(*compiler);
  selected_path(*compiler);
  selected_path(*compiler, true);
  indirect_snapshots(*compiler, true);
  faulting_return(*compiler, true);
  code_write_guard(*compiler, true);
  cpu_barrier(*compiler, true);
  std::println(
      "region backend: bounded nested calls/returns, 64-bit snapshots, "
      "fault prefixes, CP0 and code-write guards passed");
}
