#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <memory>
#include <print>
#include <span>
#include <string_view>
#include <utility>

#include <dynamips/macros.h>

import jit.amd64;
import jit.dispatch;
import jit.memory;
import jit.memory_dispatch;
import jit.mips64;

namespace {
namespace cpu = jit::mips64;
namespace mem = jit::memory;
constexpr let rwx =
    mem::permission::read | mem::permission::write | mem::permission::execute;
constexpr std::uint64_t vector_page = 0xffffffff80000000ULL;
constexpr std::uint64_t vector_pc = vector_page + 0x180;
constexpr std::array<cpu::instruction_word, 5> loop_words{
    {{0x40024800}, {0}, {0x00621023}, {0x1c40fffc}, {0}}};

fn require(bool condition, std::string_view message) -> void {
  if (!condition) {
    std::println(stderr, "count dispatch: {}", message);
    std::exit(1);
  }
}

fn same(const cpu::state &left, const cpu::state &right) -> bool {
  return left.registers == right.registers && left.pc.value == right.pc.value &&
         left.cop0.registers == right.cop0.registers &&
         left.cop0.half_cycle == right.cop0.half_cycle &&
         left.hilo.registers == right.hilo.registers &&
         left.cop1.registers == right.cop1.registers;
}

struct fixture {
  std::array<std::byte, 3 * mem::page_size> ram{};
  std::unique_ptr<mem::address_space> memory;
  std::uint64_t pc;

  explicit fixture(std::uint64_t start = 0x1100) : pc(start) {
    let created = mem::address_space::create(ram, 5);
    require(created.has_value(), "create dispatch memory");
    memory = std::move(*created);
    require(memory->map_ram({0x1000}, {0}, rwx).has_value() &&
                memory->map_ram({0x2000}, {mem::page_size}, rwx).has_value() &&
                memory->map_ram({0x3000}, {0}, rwx).has_value() &&
                memory->map_ram({0x4000}, {mem::page_size}, rwx).has_value() &&
                memory->map_ram({vector_page}, {2 * mem::page_size}, rwx)
                    .has_value(),
            "map code, physical aliases and interrupt handler");
    for (std::size_t i = 0; i < loop_words.size(); i++)
      write(pc + i * 4, loop_words[i].bits);
    write(vector_pc, 0x34140055);
  }

  fn write(std::uint64_t address, std::uint32_t bits) -> void {
    require(memory->store<std::uint32_t>({address}, bits).has_value(),
            "install guest word");
  }

  fn state(std::uint32_t count = 0, bool half = false,
             std::uint32_t target = 1000) const -> cpu::state {
    cpu::state value{};
    value.pc.value = pc;
    value.registers[2] = 0x1234;
    value.registers[3] = target;
    value.cop0.registers[9] = count;
    value.cop0.registers[11] = static_cast<std::uint32_t>(count + 10000);
    value.cop0.half_cycle = half;
    return value;
  }
};

// Compile ordinary basic blocks through the existing decoder/emitter. This
// driver has no Count-loop recognizer and observes native block exits.
fn ordinary(cpu::state &state, mem::address_space &memory,
              const jit::amd64::compiler &compiler, std::uint64_t budget,
              std::size_t block_limit) -> jit::dispatch_result {
  jit::dispatch_result result{jit::stop_reason::yielded, 0, 0, 0};
  alignas(std::max_align_t) std::array<std::byte, 64 * 1024> scratch;
  while (result.blocks < block_limit) {
    if (result.instructions == budget) {
      result.reason = jit::stop_reason::budget_exhausted;
      break;
    }
    if (state.cop0.interrupt_pending()) {
      memory.invalidate();
      state.pc = state.cop0.exception(0, state.pc, false);
    }
    const let pc = state.pc.value;
    const let remaining = budget - result.instructions;
    const let limit = std::min<std::uint64_t>(64, remaining);
    std::array<cpu::instruction_word, 65> words{};
    std::size_t count = 0;
    bool delay = false;
    while (count < limit || delay) {
      const let fetched = memory.fetch<std::uint32_t>({pc + count * 4});
      require(fetched.has_value(), "ordinary reference fetch");
      words[count++] = {*fetched};
      if (delay)
        break;
      if (cpu::has_delay_slot(words[count - 1])) {
        if (count == limit && count > 1) {
          count--;
          break;
        }
        delay = true;
      } else if (*fetched >> 26 == 0x10) {
        break;
      }
    }
    let code = jit::compile_native_block(std::span{words}.first(count), {pc},
                                          compiler, scratch, false);
    require(code.has_value(), "ordinary reference native compilation");
    if (code->instructions > remaining) {
      result.reason = jit::stop_reason::budget_exhausted;
      break;
    }
    const let distance = static_cast<std::uint32_t>(state.cop0.registers[11] -
                                                    state.cop0.registers[9]);
    const let cycles = distance ? std::uint64_t{distance} * 2 -
                                      (state.cop0.half_cycle ? 1u : 0u)
                                : UINT64_MAX;
    const let ran =
        code->code.execute(state.registers, memory, &state.cop0, &state.hilo,
                           &state.cop1, {}, remaining, 1, cycles);
    require(ran && ran->reason == jit::amd64::exit_reason::completed,
            "ordinary reference native execution");
    state.cop0.advance_cycles(ran->instructions - ran->accounted_instructions);
    state.pc.value = ran->next_pc;
    result.instructions += ran->instructions;
    result.blocks += ran->blocks;
  }
  return result;
}

fn differential() -> void {
  let compiler = jit::amd64::compiler::create();
  require(compiler.has_value(), "create ordinary reference compiler");
  for (const let cold : {std::uint16_t{0}, std::uint16_t{32}}) {
    fixture test;
    for (const bool half : {false, true})
      for (const let count : {0u, 0x7ffffffdu, 0xfffffffdu})
        for (const let delta : {0u, 1u, 2u, 5u, 17u, 0x80000000u})
          for (const let budget :
               {0u, 1u, 2u, 3u, 4u, 5u, 6u, 9u, 10u, 31u, 64u, 100u}) {
            let dispatch = jit::memory_dispatcher::create(*test.memory, cold);
            require(dispatch.has_value(), "create differential dispatcher");
            let initial = test.state(
                count, half, static_cast<std::uint32_t>(count + delta));
            let actual = initial;
            let expected = initial;
            const let result = (*dispatch)->run_slice(actual, budget, 64);
            const let reference =
                ordinary(expected, *test.memory, *compiler, budget, 64);
            require(result && same(actual, expected) &&
                        result->instructions == reference.instructions &&
                        result->blocks == reference.blocks &&
                        result->reason == reference.reason,
                    "batched dispatcher matches ordinary native blocks at "
                    "exact budgets");
          }
    for (const let blocks : {0u, 1u, 2u, 3u, 4u, 5u, 63u, 64u}) {
      let dispatch = jit::memory_dispatcher::create(*test.memory, cold);
      require(dispatch.has_value(), "create block-limit dispatcher");
      let actual = test.state();
      let expected = actual;
      const let result = (*dispatch)->run_slice(actual, UINT64_MAX, blocks);
      const let reference =
          ordinary(expected, *test.memory, *compiler, UINT64_MAX, blocks);
      require(
          result && same(actual, expected) &&
              result->instructions == reference.instructions &&
              result->blocks == reference.blocks,
          "full iterations preserve two original blocks per five instructions");
    }
  }
}

fn timer_prefix() -> void {
  fixture test;
  let dispatch = jit::memory_dispatcher::create(*test.memory);
  let compiler = jit::amd64::compiler::create();
  require(dispatch && compiler, "create precise timer execution");
  let actual = test.state();
  actual.cop0.registers[11] = 3;
  actual.cop0.registers[12] = 0x8001;
  let expected = actual;
  const let result = (*dispatch)->run_slice(actual, 7, 4);
  const let reference = ordinary(expected, *test.memory, *compiler, 7, 4);
  require(
      result && same(actual, expected) && result->instructions == 7 &&
          result->blocks == 4 && reference.instructions == 7 &&
          reference.blocks == 4 && actual.cop0.registers[14] == test.pc + 4 &&
          actual.pc.value == vector_pc + 4 && actual.registers[20] == 0x55 &&
          (actual.cop0.registers[12] & 2) != 0 &&
          (actual.cop0.registers[13] & 0x7c) == 0,
      "Compare after six cycles traps before loop tail with EPC at pc+4");
}

fn alias_edits() -> void {
  let compiler = jit::amd64::compiler::create();
  require(compiler.has_value(), "create alias reference compiler");
  constexpr std::array mutations{0x40044800u, 0x25080001u, 0x24020000u,
                                 0x1840fffcu, 0x25290001u};
  for (std::size_t changed = 0; changed < loop_words.size(); changed++) {
    fixture test{0x1ff8};
    let dispatch = jit::memory_dispatcher::create(*test.memory);
    require(dispatch.has_value(), "create cross-page alias dispatcher");
    let warm = test.state();
    const let primed = (*dispatch)->run_slice(warm, 10, 4);
    require(primed && primed->instructions == 10 && primed->blocks == 4 &&
                warm.pc.value == test.pc,
            "warm a loop whose identity crosses executable pages");
    test.write(test.pc + 0x2000 + changed * 4, mutations[changed]);
    let actual = test.state();
    let expected = actual;
    const let result = (*dispatch)->run_slice(actual, 10, 64);
    const let reference = ordinary(expected, *test.memory, *compiler, 10, 64);
    require(
        result && same(actual, expected) && result->compiled != 0 &&
            result->instructions == reference.instructions &&
            result->blocks == reference.blocks,
        "alias edits to every watched word replace the cached Count pattern");
    if (changed == 1 || changed == 4)
      require(actual.registers[changed == 1 ? 8 : 9] != 0,
              "new side effects in either formerly empty slot execute");
  }
}

fn speculative_backing() -> void {
  for (const bool prime : {false, true})
    for (const bool remove : {false, true}) {
      fixture test{0x1ff8};
      let dispatch = jit::memory_dispatcher::create(*test.memory);
      require(dispatch.has_value(), "create speculative backing dispatcher");
      if (prime) {
        let warm = test.state();
        require((*dispatch)->run_slice(warm, 10, 4).has_value(),
                "warm complete source identity before editing mappings");
      }
      require(remove ? test.memory->unmap({0x2000}).has_value()
                     : test.memory
                           ->protect({0x2000}, mem::permission::read |
                                                   mem::permission::write)
                           .has_value(),
              "remove second page or revoke its execution permission");
      let state = test.state(7);
      state.cop0.registers[8] = 0xfeed;
      state.cop0.registers[14] = 0xbeef;
      const let result = (*dispatch)->run_slice(state, 1, 64);
      require(result && result->instructions == 1 && result->blocks == 1 &&
                  state.pc.value == test.pc + 4 && state.registers[2] == 7 &&
                  state.cop0.registers[9] == 7 && state.cop0.half_cycle &&
                  state.cop0.registers[8] == 0xfeed &&
                  state.cop0.registers[14] == 0xbeef,
              "speculative second-page proof does not fault before the valid "
              "Count word");
    }
}

fn cpu_aliased_source() -> void {
  struct alignas(mem::page_size) storage {
    cpu::state guest{};
    std::array<std::byte, mem::page_size> extra{};
  } backing;

  let bytes = std::as_writable_bytes(std::span{&backing, 1});
  const let source = std::as_writable_bytes(std::span{backing.guest.registers})
                         .subspan(20 * sizeof(std::uint64_t), 20);
  const let offset = reinterpret_cast<std::uintptr_t>(source.data()) -
                     reinterpret_cast<std::uintptr_t>(bytes.data());
  require(offset % 4 == 0 && offset + source.size() <= mem::page_size,
          "CPU-aliased source fits the first executable RAM page");
  for (std::size_t i = 0; i < loop_words.size(); i++)
    for (unsigned byte = 0; byte < 4; byte++)
      source[i * 4 + byte] =
          static_cast<std::byte>(loop_words[i].bits >> (24 - byte * 8));
  let &actual = backing.guest;
  actual.pc = {0x1000 + offset};
  actual.registers[3] = 1000;
  actual.cop0.registers[11] = 10000;
  let expected = actual;
  let created = mem::address_space::create(bytes, 1);
  require(created.has_value(), "create RAM containing CPU state");
  let memory = std::move(*created);
  require(memory->map_ram({0x1000}, {0}, rwx).has_value(),
          "map a Count loop whose backing aliases the CPU object");
  let dispatch = jit::memory_dispatcher::create(*memory);
  let compiler = jit::amd64::compiler::create();
  require(dispatch && compiler, "create CPU-source-alias execution paths");
  const let result = (*dispatch)->run_slice(actual, 10, 4);
  const let reference = ordinary(expected, *memory, *compiler, 10, 4);
  require(result && result->compiled == 2 && result->instructions == 10 &&
              result->blocks == 4 && reference.instructions == 10 &&
              same(actual, expected),
          "CPU-aliased code declines batching and compiles both ordinary "
          "blocks even when this particular loop leaves its source unchanged");
}

struct future_translation {
  std::byte *tail;
  cpu::state *guest;
  unsigned calls = 0;
  bool mutates_guest = false;

  static fn resolve(void *context, mem::virtual_address where,
                     mem::access_kind access, mem::access_size size) noexcept
      -> std::expected<mem::translated_page, mem::fault> {
    let &self = *static_cast<future_translation *>(context);
    if ((where.value & ~std::uint64_t{mem::page_size - 1}) != 0x2000 ||
        access != mem::access_kind::execute)
      return std::unexpected(
          mem::fault{where, mem::fault_reason::unmapped, access, size});
    self.calls++;
    if (self.mutates_guest)
      self.guest->registers[5] = 0x55;
    return mem::translated_page{
        {mem::page_size + (where.value & (mem::page_size - 1))},
        self.tail,
        nullptr};
  }
};

fn speculative_translation_effects() -> void {
  for (const bool preserving : {false, true})
    for (const bool primed : {false, true}) {
      std::array<std::byte, 2 * mem::page_size> ram{};
      for (std::size_t i = 0; i < loop_words.size(); i++)
        for (unsigned byte = 0; byte < 4; byte++)
          ram[mem::page_size - 4 + i * 4 + byte] =
              static_cast<std::byte>(loop_words[i].bits >> (24 - byte * 8));
      cpu::state state{};
      state.pc = {0x1ffc};
      state.registers[3] = 1000;
      state.registers[5] = 0x11;
      state.cop0.registers[9] = 7;
      state.cop0.registers[11] = 10000;
      const let initial = state;
      future_translation translation{ram.data() + mem::page_size, &state};
      let created = mem::address_space::create(ram, 1);
      require(created.has_value(), "allocate future Count-loop source pages");
      let memory = std::move(*created);
      require(memory->map_ram({0x1000}, {0}, rwx).has_value() &&
                  memory->may_speculate_translation(),
              "static mappings allow effect-free speculative lookup");
      let dispatcher = jit::memory_dispatcher::create(*memory);
      require(dispatcher.has_value(), "create effect-gated Count dispatcher");
      if (primed) {
        memory->set_translation(
            &translation, &future_translation::resolve,
            mem::translation_effect::guest_state_preserving);
        const let warm = (*dispatcher)->run_slice(state, 1, 64);
        require(
            warm && warm->instructions == 1 && translation.calls == 1 &&
                state.registers[5] == 0x11,
            "prime the cached five-word proof through a preserving callback");
        state = initial;
      }
      translation.mutates_guest = !preserving;
      if (preserving)
        memory->set_translation(
            &translation, &future_translation::resolve,
            mem::translation_effect::guest_state_preserving);
      else
        memory->set_translation(&translation, &future_translation::resolve);
      require(
          memory->may_speculate_translation() == preserving,
          "only explicit preserving translation permits future-page lookup");
      const let before = translation.calls;
      const let first = (*dispatcher)->run_slice(state, 1, 64);
      require(first && first->instructions == 1 && first->blocks == 1 &&
                  state.pc.value == 0x2000 && state.registers[2] == 7 &&
                  state.registers[5] == 0x11 &&
                  translation.calls == before + (preserving ? 1u : 0u),
              "a one-slot Count budget never invokes a future opaque callback, "
              "including after replacing a cached preserving proof");
      const let actual_tail = (*dispatcher)->run_slice(state, 1, 64);
      require(actual_tail && actual_tail->instructions == 1 &&
                  state.pc.value == 0x2004 && translation.calls == before + 1 &&
                  state.registers[5] == (preserving ? 0x11u : 0x55u),
              "the actual next instruction invokes opaque translation, while "
              "preserving speculative translation changes diagnostics only");
    }
}
fn opaque_hot_loop() -> void {
  std::array<std::byte, 2 * mem::page_size> ram{};
  cpu::state state{};
  state.pc = {0x1ff0};
  state.registers[5] = 0x11;
  future_translation translation{ram.data() + mem::page_size, &state, 0, true};
  let created = mem::address_space::create(ram, 1);
  require(created.has_value(), "allocate an opaque hot-loop fixture");
  let memory = std::move(*created);
  require(memory->map_ram({0x1000}, {0}, rwx).has_value(),
          "map the complete hot-loop block");
  memory->set_translation(&translation, &future_translation::resolve);
  constexpr std::array program{0x24210001u, 0x080007fcu, 0u};
  for (std::size_t i = 0; i < program.size(); i++)
    require(
        memory->store<std::uint32_t>({0x1ff0 + i * 4}, program[i]).has_value(),
        "install a loop that never reaches the translated next page");
  let dispatch = jit::memory_dispatcher::create(*memory);
  require(dispatch.has_value(), "create an opaque hot-loop dispatcher");
  const let result = (*dispatch)->run_slice(state, 4096 * 3, 4096);
  require(result && result->instructions == 4096 * 3 &&
              result->blocks == 4096 && state.pc.value == 0x1ff0 &&
              state.registers[1] == 4096 && state.registers[5] == 0x11 &&
              translation.calls == 0,
          "hot trace discovery cannot invoke an untaken opaque translation");
}

fn retire_cached_speculative_trace() -> void {
  std::array<std::byte, 3 * mem::page_size> ram{};
  cpu::state state{};
  state.pc = {0x1ff0};
  state.registers[5] = 0x11;
  future_translation translation{ram.data() + mem::page_size, &state};
  let made = mem::address_space::create(ram, 2);
  require(made.has_value(), "allocate a cached speculative trace fixture");
  let memory = std::move(*made);
  require(memory->map_ram({0x1000}, {0}, rwx).has_value() &&
              memory->map_ram({0x3000}, {2 * mem::page_size}, rwx).has_value(),
          "map both pages on the actual execution path");
  memory->set_translation(&translation, &future_translation::resolve,
                          mem::translation_effect::guest_state_preserving);
  for (const auto [address, word] :
       std::array{std::pair{0xff0u, 0x10000403u}, std::pair{0xff4u, 0u},
                  std::pair{0x1000u, 0x1000fffbu}, std::pair{0x1004u, 0u},
                  std::pair{0x2000u, 0x080007fcu}, std::pair{0x2004u, 0u}}) {
    std::array<std::byte, 4> encoded{};
    for (unsigned byte = 0; byte < 4; byte++)
      encoded[byte] = static_cast<std::byte>(word >> (24 - byte * 8));
    require(memory->write_physical({address}, encoded).has_value(),
            "install a trace with an untaken speculative backedge");
  }
  let dispatch = jit::memory_dispatcher::create(*memory);
  require(dispatch.has_value(), "create a speculative trace dispatcher");
  const let warm = (*dispatch)->run_slice(state, 100000, 4096);
  require(warm && warm->compiled >= 3 && state.pc.value == 0x1ff0 &&
              translation.calls != 0 && state.registers[5] == 0x11,
          "a preserving callback may supply untaken cached trace words");
  const let calls = translation.calls;
  translation.mutates_guest = true;
  memory->set_translation(&translation, &future_translation::resolve);
  const let resumed = (*dispatch)->run_slice(state, 100, 1);
  require(resumed && resumed->instructions == 2 && resumed->compiled == 1 &&
              state.pc.value == 0x3000 && state.registers[5] == 0x11 &&
              translation.calls == calls,
          "an opaque callback retires cached speculative source before any "
          "future-page revalidation");
}

fn fragmented_tail_budgets() -> void {
  for (const std::uint16_t cold : {std::uint16_t{0}, std::uint16_t{32}})
    for (const bool half : {false, true}) {
      fixture f;
      let state = f.state(0xfffffff8u, half, 1000);
      let dispatch = jit::memory_dispatcher::create(*f.memory, cold);
      require(dispatch.has_value(), "create a persistent small-budget cache");
      const let first = (*dispatch)->run_slice(state, 2, 64);
      require(first && first->instructions == 2 && first->blocks == 2 &&
                  state.pc.value == f.pc + 8,
              "a two-slot slice caches Count and a separate one-slot NOP tail");
      const let second = (*dispatch)->run_slice(state, 3, 64);
      require(second && second->instructions == 3 && second->blocks == 1 &&
                  state.pc.value == f.pc,
              "the remaining SUBU and branch form a third cached block");
      const let sample = static_cast<std::uint32_t>(state.cop0.registers[9]);
      let expected_clock = state.cop0;
      expected_clock.advance_cycles(2);
      const let bounded = (*dispatch)->run_slice(state, 5, 2);
      require(bounded && bounded->instructions == 2 && bounded->blocks == 2 &&
                  state.pc.value == f.pc + 8 &&
                  state.registers[2] ==
                      static_cast<std::uint64_t>(
                          std::int64_t{std::bit_cast<std::int32_t>(sample)}) &&
                  state.cop0.registers == expected_clock.registers &&
                  state.cop0.half_cycle == expected_clock.half_cycle,
              "batching preserves an existing fragmented cache's block limit");
      const let finish = (*dispatch)->run_slice(state, 3, 1);
      require(finish && finish->instructions == 3 && finish->blocks == 1 &&
                  state.pc.value == f.pc,
              "later slices resume the existing three-block iteration");
    }
}

} // namespace

fn main() -> int {
  differential();
  timer_prefix();
  alias_edits();
  speculative_backing();
  speculative_translation_effects();
  cpu_aliased_source();
  opaque_hot_loop();
  retire_cached_speculative_trace();
  fragmented_tail_budgets();
  std::println(
      "count dispatch: native differential, exact budgets/blocks, timer EPC, "
      "all-word aliases and missing speculative pages passed");
}
