#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <print>
#include <span>
#include <string_view>

#include <dynamips/macros.h>

import jit.count_batch;
import jit.mips64;

namespace {
namespace batch = jit::count_batch;
namespace cpu = jit::mips64;

fn require(bool condition, std::string_view message) -> void {
  if (!condition) {
    std::println(stderr, "count batch: {}", message);
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

fn program(batch::loop pattern) -> std::array<cpu::instruction_word, 5> {
  return {{{0x40004800u | (std::uint32_t{pattern.count} << 16)},
           {0},
           {(std::uint32_t{pattern.target} << 21) |
            (std::uint32_t{pattern.count} << 16) |
            (std::uint32_t{pattern.difference} << 11) | 0x23u},
           {(7u << 26) | (std::uint32_t{pattern.difference} << 21) | 0xfffcu},
           {0}}};
}

// The reference walks individual instructions and clock edges. A speculative
// iteration is discarded if a Compare edge would precede its last slot.
fn tick(cpu::cop0_state &clock) -> bool {
  const let before = static_cast<std::uint32_t>(clock.registers[9]);
  const bool increment = clock.half_cycle;
  clock.half_cycle = !clock.half_cycle;
  const let after = static_cast<std::uint32_t>(before + (increment ? 1u : 0u));
  clock.registers[9] = after;
  const bool matched =
      increment && after == static_cast<std::uint32_t>(clock.registers[11]);
  if (matched)
    clock.registers[13] |= 1u << 15;
  return matched;
}

fn reference(batch::loop pattern, cpu::state &state, std::uint64_t budget,
               std::size_t blocks) -> std::optional<batch::result> {
  batch::result result{};
  while (budget >= 5 && blocks >= 2 && !state.cop0.interrupt_pending()) {
    let next = state;
    const let sample = static_cast<std::uint32_t>(next.cop0.registers[9]);
    next.registers[pattern.count] = static_cast<std::uint64_t>(
        std::int64_t{std::bit_cast<std::int32_t>(sample)});
    bool early_compare = tick(next.cop0);
    early_compare = tick(next.cop0) || early_compare;
    const let target =
        pattern.target
            ? static_cast<std::uint32_t>(next.registers[pattern.target])
            : 0u;
    const let difference = std::bit_cast<std::int32_t>(target - sample);
    next.registers[pattern.difference] =
        static_cast<std::uint64_t>(std::int64_t{difference});
    early_compare = tick(next.cop0) || early_compare;
    const bool taken = difference > 0;
    early_compare = tick(next.cop0) || early_compare;
    const bool final_compare = tick(next.cop0);
    if (early_compare)
      break;
    if (!taken)
      next.pc.value += 20;
    state = next;
    result.instructions += 5;
    result.blocks += 2;
    result.iterations++;
    budget -= 5;
    blocks -= 2;
    if (!taken || final_compare)
      break;
  }
  return result.iterations ? std::optional{result} : std::nullopt;
}

fn compare(batch::loop pattern, const cpu::state &initial,
             std::uint64_t budget, std::size_t blocks) -> void {
  let actual = initial;
  let expected = initial;
  const let fast = batch::run(pattern, actual, budget, blocks);
  const let slow = reference(pattern, expected, budget, blocks);
  require(fast.has_value() == slow.has_value(),
          "batch availability matches reference");
  require(same(actual, expected),
          "all architectural fields match iterative execution");
  if (fast) {
    require(fast->instructions == slow->instructions &&
                fast->blocks == slow->blocks &&
                fast->iterations == slow->iterations &&
                fast->instructions <= budget && fast->blocks <= blocks,
            "exact executed slot and two-block iteration accounting");
  }
}

fn decoding() -> void {
  for (const let pattern : std::array{
           batch::loop{2, 3, 2}, batch::loop{5, 6, 7}, batch::loop{9, 0, 10}}) {
    const let words = program(pattern);
    const let decoded = batch::decode(words);
    require(decoded && decoded->count == pattern.count &&
                decoded->target == pattern.target &&
                decoded->difference == pattern.difference,
            "decode canonical Count loop with shared or separate destinations");
    for (unsigned bit = 0; bit < 32; bit++) {
      let changed = words;
      changed[1].bits ^= std::uint32_t{1} << bit;
      require(!batch::decode(changed), "reject effects replacing first NOP");
      changed = words;
      changed[4].bits ^= std::uint32_t{1} << bit;
      require(!batch::decode(changed), "reject effects replacing delay NOP");
    }
    for (unsigned index : {0u, 2u, 3u}) {
      for (const let mask : {0x40000000u, 0x00000001u, 0x00000040u}) {
        let changed = words;
        changed[index].bits ^= mask;
        require(!batch::decode(changed),
                "reject unsupported opcodes and fields");
      }
    }
    require(!batch::decode(std::span{words}.first(4)),
            "reject incomplete loop");
  }
  for (const let invalid :
       std::array{batch::loop{0, 3, 2}, batch::loop{2, 3, 0},
                  batch::loop{2, 2, 3}, batch::loop{2, 3, 3}})
    require(!batch::decode(program(invalid)),
            "reject zero destinations and changing targets");
}

fn boundaries() -> void {
  constexpr std::array counts{0u,          1u,          2u,
                              0x7fffffffu, 0x80000000u, 0xfffffffdu,
                              0xfffffffeu, 0xffffffffu};
  constexpr std::array deltas{
      0u, 1u,  2u,          3u,          4u,          5u,
      6u, 31u, 0x7fffffffu, 0x80000000u, 0xfffffffdu, 0xffffffffu};
  for (const bool half : {false, true})
    for (const let count : counts)
      for (const let delta : deltas)
        for (const let deadline : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 17u, 255u})
          for (const let pattern :
               {batch::loop{2, 3, 2}, batch::loop{5, 6, 7}}) {
            cpu::state state{};
            state.pc.value = 0xffffffff80000ff0ULL;
            state.cop0.registers[9] = count;
            state.cop0.registers[11] =
                static_cast<std::uint32_t>(count + deadline);
            state.cop0.registers[12] = 0x8001;
            state.cop0.half_cycle = half;
            state.registers[pattern.target] =
                static_cast<std::uint32_t>(count + delta);
            for (const let budget :
                 {0u, 1u, 4u, 5u, 6u, 9u, 10u, 11u, 99u, 100u})
              compare(pattern, state, budget, 64);
            for (const let blocks : {0u, 1u, 2u, 3u, 4u, 5u, 63u, 64u})
              compare(pattern, state, UINT64_MAX, blocks);
          }
}

fn seeded() -> void {
  std::uint64_t seed = 0xbeac4c08cdef9127ULL;
  const let random = [&seed]() {
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    return seed;
  };
  for (unsigned trial = 0; trial < 20000; trial++) {
    cpu::state state{};
    state.pc.value = 0x60000000 | (random() & 0x0ffffffc);
    for (let &value : state.registers)
      value = random();
    for (let &value : state.cop0.registers)
      value = random();
    state.cop0.registers[12] = random() & 1;
    state.cop0.registers[13] &= ~std::uint64_t{0xff00};
    state.cop0.registers[22] &= ~std::uint64_t{0x400};
    state.cop0.half_cycle = (random() & 1) != 0;
    if (trial & 1)
      state.cop0.registers[11] =
          static_cast<std::uint32_t>(state.cop0.registers[9] + random() % 200);
    const let pattern = trial % 3 == 0   ? batch::loop{9, 0, 10}
                        : trial % 3 == 1 ? batch::loop{2, 3, 2}
                                         : batch::loop{5, 6, 7};
    if (pattern.target && (trial & 2))
      state.registers[pattern.target] =
          static_cast<std::uint32_t>(state.cop0.registers[9] + random() % 256);
    compare(pattern, state, random() % 1000,
            static_cast<std::size_t>(random() % 128));
  }
}

fn rejection() -> void {
  cpu::state state{};
  state.pc.value = 0x60001000;
  state.cop0.registers[11] = 1000;
  state.registers[3] = 100;
  const let unchanged = [](const cpu::state &input, batch::loop pattern) {
    let output = input;
    require(!batch::run(pattern, output, 100, 64) && same(output, input),
            "rejected execution leaves architectural state unchanged");
  };
  for (const let pattern :
       {batch::loop{0, 3, 2}, batch::loop{2, 3, 0}, batch::loop{2, 2, 3},
        batch::loop{2, 3, 3}, batch::loop{32, 3, 2}, batch::loop{2, 32, 3},
        batch::loop{2, 3, 32}})
    unchanged(state, pattern);
  for (const let status : {8u, 16u, 24u, 0x10000010u}) {
    let bad = state;
    bad.cop0.registers[12] = status;
    unchanged(bad, {2, 3, 2});
  }
  let bad = state;
  bad.pc.value++;
  unchanged(bad, {2, 3, 2});
  bad = state;
  bad.pc.value = UINT64_MAX - 3;
  unchanged(bad, {2, 3, 2});
  bad = state;
  bad.cop0.registers[22] = 0x400;
  unchanged(bad, {2, 3, 2});
  bad = state;
  bad.cop0.registers[12] = 0x8001;
  bad.cop0.registers[13] = 0x8000;
  unchanged(bad, {2, 3, 2});
}
} // namespace

fn main() -> int {
  decoding();
  boundaries();
  seeded();
  rejection();
  std::println("count batch: decode, seeded iterative equivalence, wrap, "
               "budgets and timer edges passed");
}
