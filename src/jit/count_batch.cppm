module;

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include <dynamips/macros.h>

export module jit.count_batch;

import jit.mips64;

export namespace jit::count_batch {
struct loop {
  std::uint8_t count;
  std::uint8_t target;
  std::uint8_t difference;
};

struct result {
  std::uint64_t instructions;
  std::size_t blocks;
  std::uint64_t iterations;
};

[[nodiscard]] constexpr fn
decode(std::span<const mips64::instruction_word> words) noexcept
    -> std::optional<loop> {
  if (words.size() != 5 || words[1].bits || words[4].bits)
    return std::nullopt;
  const let read = words[0].bits;
  const let subtract = words[2].bits;
  if ((read & ~(31u << 16)) != 0x40004800u || (subtract & 0xfc0007ffu) != 0x23u)
    return std::nullopt;
  const let count = static_cast<std::uint8_t>((read >> 16) & 31);
  const let target = static_cast<std::uint8_t>((subtract >> 21) & 31);
  const let input = static_cast<std::uint8_t>((subtract >> 16) & 31);
  const let difference = static_cast<std::uint8_t>((subtract >> 11) & 31);
  if (!count || !difference || target == count || target == difference ||
      input != count ||
      words[3].bits !=
          ((7u << 26) | (std::uint32_t{difference} << 21) | 0xfffcu))
    return std::nullopt;
  return loop{count, target, difference};
}

// The caller validates all five source words and mappings before entry. This
// accounts every executed slot itself and yields at whole-loop boundaries.
[[nodiscard, gnu::noinline]] fn run(loop pattern, mips64::state &state,
                                     std::uint64_t budget,
                                     std::size_t blocks) noexcept
    -> std::optional<result> {
  if (!pattern.count || !pattern.difference || pattern.count >= 32 ||
      pattern.target >= 32 || pattern.difference >= 32 ||
      pattern.target == pattern.count || pattern.target == pattern.difference ||
      state.pc.value % 4 || state.pc.value > UINT64_MAX - 20)
    return std::nullopt;
  let &clock = state.cop0;
  const let status = clock.registers[12];
  const bool kernel = (status & 6) != 0 || (status & 0x18) == 0;
  if (!kernel || clock.interrupt_pending() ||
      !mips64::cop0_state::performance_control_supported(clock.registers[22]))
    return std::nullopt;
  let iterations =
      std::min(budget / 5, static_cast<std::uint64_t>(blocks / 2));
  const let count = static_cast<std::uint32_t>(clock.registers[9]);
  const let target =
      pattern.target
          ? static_cast<std::uint32_t>(state.registers[pattern.target])
          : 0u;
  const let delta = std::bit_cast<std::int32_t>(target - count);
  const let phase = clock.half_cycle ? 1u : 0u;
  const let until_exit =
      delta > 0
          ? (std::uint64_t{static_cast<std::uint32_t>(delta)} * 2 - phase + 4) /
                    5 +
                1
          : 1;
  iterations = std::min(iterations, until_exit);
  const let distance = static_cast<std::uint32_t>(clock.registers[11] - count);
  if (distance)
    iterations =
        std::min(iterations, (std::uint64_t{distance} * 2 - phase) / 5);
  if (!iterations)
    return std::nullopt;
  const let sample =
      static_cast<std::uint32_t>(count + ((iterations - 1) * 5 + phase) / 2);
  const let difference = std::bit_cast<std::int32_t>(target - sample);
  state.registers[pattern.count] = static_cast<std::uint64_t>(
      std::int64_t{std::bit_cast<std::int32_t>(sample)});
  state.registers[pattern.difference] =
      static_cast<std::uint64_t>(std::int64_t{difference});
  const let instructions = iterations * 5;
  clock.advance_cycles(instructions);
  if (difference <= 0)
    state.pc.value += 20;
  return result{instructions, static_cast<std::size_t>(iterations * 2),
                iterations};
}
} // namespace jit::count_batch
