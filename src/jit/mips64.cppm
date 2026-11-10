module;

#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <memory_resource>
#include <print>
#include <span>
#include <utility>

#include <dynamips/macros.h>

export module jit.mips64;

import jit.ir;

export namespace jit::mips64 {

struct instruction_word {
  std::uint32_t bits;
};

struct guest_address {
  std::uint64_t value;
};

constexpr fn is_branch(instruction_word word) noexcept -> bool {
  const let op = word.bits >> 26;
  return op == 4 || op == 5;
}

struct state {
  std::array<std::uint64_t, 32> registers{};
  guest_address pc{};

  fn print() const -> void {
    std::println("; native execution, PC={:#x}", pc.value);
    for (std::size_t i = 0; i < registers.size(); i++)
      std::println("r{} = {:#x}", i, registers[i]);
  }
};

enum class error_code {
  unaligned_pc,
  block_too_large,
  unsupported_instruction,
  invalid_encoding,
  out_of_memory,
  invalid_ir,
  missing_delay_slot,
  branch_in_delay_slot,
};

struct error {
  error_code code;
  std::size_t instruction_index;
};

[[nodiscard]] fn translate(std::span<const instruction_word> words,
                            guest_address start,
                            std::pmr::memory_resource &memory) noexcept
    -> std::expected<ir::block, error> {
  if (start.value % 4 != 0)
    return std::unexpected(error{error_code::unaligned_pc, 0});
  for (std::size_t i = 0; i < words.size(); i++) {
    if (!is_branch(words[i]))
      continue;
    if (i + 1 == words.size())
      return std::unexpected(error{error_code::missing_delay_slot, i});
    if (is_branch(words[i + 1]))
      return std::unexpected(error{error_code::branch_in_delay_slot, i + 1});
    words = words.first(i + 2);
    break;
  }
  constexpr std::size_t nodes_per_instruction = 4;
  const let memory_instructions = static_cast<std::size_t>(
      std::ranges::count_if(words, [](instruction_word word) {
        const let op = word.bits >> 26;
        return op == 0x23 || op == 0x2b || op == 0x37 || op == 0x3f;
      }));
  if (memory_instructions > std::numeric_limits<std::uint32_t>::max() ||
      words.size() >
          (std::numeric_limits<std::uint32_t>::max() - memory_instructions) /
              nodes_per_instruction ||
      words.size() >
          (std::numeric_limits<std::uint64_t>::max() - start.value) / 4)
    return std::unexpected(error{error_code::block_too_large, 0});
  let created = ir::block::create(
      {start.value, static_cast<std::uint32_t>(words.size()), 32},
      words.size() * nodes_per_instruction + memory_instructions, memory);
  if (!created)
    return std::unexpected(
        error{created.error().code == ir::error_code::out_of_memory
                  ? error_code::out_of_memory
                  : error_code::block_too_large,
              0});
  let result = std::move(*created);

  std::array<ir::value_id, 32> registers{};
  for (std::size_t i = 0; i < words.size(); i++) {
    const let bits = words[i].bits;
    if (bits == 0)
      continue;
    const let opcode = bits >> 26;
    const let rs = static_cast<std::uint16_t>((bits >> 21) & 31);
    const let rt = static_cast<std::uint16_t>((bits >> 16) & 31);
    const let rd = static_cast<std::uint16_t>((bits >> 11) & 31);
    const let function = bits & 63;
    const bool immediate = opcode == 0x0d;
    const bool load = opcode == 0x23 || opcode == 0x37;
    const bool store = opcode == 0x2b || opcode == 0x3f;
    const bool memory_access = load || store;
    const bool branch = is_branch(words[i]);
    if (!immediate && !memory_access && !branch &&
        !(opcode == 0 && (function == 0x26 || function == 0x2d)))
      return std::unexpected(error{error_code::unsupported_instruction, i});
    if (!immediate && !memory_access && !branch && ((bits >> 6) & 31) != 0)
      return std::unexpected(error{error_code::invalid_encoding, i});
    const let destination = immediate || memory_access ? rt : rd;
    if (destination == 0 && !memory_access && !branch)
      continue;
    const let source = static_cast<std::uint32_t>(i);
    const let failed = [&](ir::error failure) {
      return std::unexpected(error{failure.code == ir::error_code::out_of_memory
                                       ? error_code::out_of_memory
                                       : error_code::invalid_ir,
                                   i});
    };
    const let read =
        [&](std::uint16_t index) -> std::expected<ir::value_id, ir::error> {
      if (registers[index].index == std::numeric_limits<std::uint32_t>::max()) {
        let value =
            result.append(index == 0 ? ir::node::constant(0, source)
                                     : ir::node::read({index}, source));
        if (!value)
          return std::unexpected(value.error());
        registers[index] = *value;
      }
      return registers[index];
    };
    const let left = read(rs);
    if (!left)
      return failed(left.error());
    if (branch) {
      const let right = read(rt);
      if (!right)
        return failed(right.error());
      const let displacement =
          std::int64_t{static_cast<std::int16_t>(bits & 0xffff)} * 4;
      const let target =
          start.value + i * 4 + 4 + static_cast<std::uint64_t>(displacement);
      if (let exit =
              result.set_branch({*left, *right, target, source,
                                 opcode == 4 ? ir::branch_kind::equal
                                             : ir::branch_kind::not_equal});
          !exit)
        return failed(exit.error());
      continue;
    }
    if (memory_access) {
      const let offset = result.append(
          ir::node::constant(static_cast<std::uint64_t>(std::int64_t{
                                 static_cast<std::int16_t>(bits & 0xffff)}),
                             source));
      if (!offset)
        return failed(offset.error());
      const let address = result.append(
          ir::node::binary(ir::binary_kind::add_wrap, *left, *offset, source));
      if (!address)
        return failed(address.error());
      const let size = opcode == 0x23 || opcode == 0x2b ? ir::width::bits32
                                                        : ir::width::bits64;
      if (store) {
        const let value = read(rt);
        if (!value)
          return failed(value.error());
        if (let written =
                result.append(ir::node::store(*address, *value, source, size));
            !written)
          return failed(written.error());
      } else {
        const let value = result.append(ir::node::load(*address, source, size));
        if (!value)
          return failed(value.error());
        if (destination != 0) {
          if (let written =
                  result.append(ir::node::write({destination}, *value, source));
              !written)
            return failed(written.error());
          registers[destination] = *value;
        }
      }
      continue;
    }
    const let right =
        immediate ? result.append(ir::node::constant(bits & 0xffff, source))
                  : read(rt);
    if (!right)
      return failed(right.error());
    const let operation = immediate          ? ir::binary_kind::bit_or
                          : function == 0x26 ? ir::binary_kind::bit_xor
                                             : ir::binary_kind::add_wrap;
    const let value =
        result.append(ir::node::binary(operation, *left, *right, source));
    if (!value)
      return failed(value.error());
    if (let written =
            result.append(ir::node::write({destination}, *value, source));
        !written)
      return failed(written.error());
    registers[destination] = *value;
  }
  return result;
}

} // namespace jit::mips64
