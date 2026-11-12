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
  return op == 4 || op == 5 || op == 0x14 || op == 0x15 ||
         (op == 1 && ((word.bits >> 16) & 31) <= 3);
}

constexpr fn has_delay_slot(instruction_word word) noexcept -> bool {
  return is_branch(word) || word.bits >> 26 == 2 || word.bits >> 26 == 3 ||
         (word.bits >> 26 == 0 &&
          ((word.bits & 63) == 8 || (word.bits & 63) == 9));
}

constexpr fn jump_target(instruction_word word, guest_address pc) noexcept
    -> guest_address {
  const std::uint64_t low = std::uint64_t{word.bits & 0x03ff'ffffu} << 2;

  return {
      ((pc.value + 4) & ~std::uint64_t{0x0fff'ffff}) | low,
  };
}

struct cop0_state {
  std::array<std::uint64_t, 32> registers{};

  fn syscall(guest_address pc, bool delay_slot) noexcept -> guest_address {
    let &status = registers[12];
    let &cause = registers[13];
    cause = (cause & ~std::uint64_t{0x7c}) | (8u << 2);
    if (!(status & 2)) {
      registers[14] = pc.value;
      cause = (cause & ~(std::uint64_t{1} << 31)) |
              (delay_slot ? std::uint64_t{1} << 31 : 0);
    }
    status |= 2;
    return {status & (1u << 22) ? 0xffffffffbfc00380ULL
                                : 0xffffffff80000180ULL};
  }

  fn write(std::uint16_t slot, std::uint32_t value) noexcept -> void {
    if (slot == 8 || slot == 15 || slot == 16)
      return;
    registers[slot] = value;
    if (slot == 11)
      registers[13] &= ~(1u << 15);
  }
};

struct hilo_state {
  std::array<std::uint64_t, 2> registers{};
};

struct state {
  std::array<std::uint64_t, 32> registers{};
  guest_address pc{};
  cop0_state cop0{};
  hilo_state hilo{};

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
    if ((words[i].bits >> 26) == 0 && (words[i].bits & 63) == 0x0c) {
      words = words.first(i + 1);
      break;
    }
    if (!has_delay_slot(words[i]))
      continue;
    if (i + 1 == words.size())
      return std::unexpected(error{error_code::missing_delay_slot, i});
    if (has_delay_slot(words[i + 1]))
      return std::unexpected(error{error_code::branch_in_delay_slot, i + 1});
    words = words.first(i + 2);
    break;
  }
  constexpr std::size_t nodes_per_instruction = 4;
  const let extra_nodes = static_cast<std::size_t>(
      std::ranges::count_if(words, [](instruction_word word) {
        const let op = word.bits >> 26;
        return (op == 0 && (word.bits & 63) == 0x21) || op == 0x09 ||
               op == 0x20 || op == 0x23 || op == 0x28 || op == 0x2b ||
               op == 0x37 || op == 0x3f;
      }));
  if (extra_nodes > std::numeric_limits<std::uint32_t>::max() ||
      words.size() > (std::numeric_limits<std::uint32_t>::max() - extra_nodes) /
                         nodes_per_instruction ||
      words.size() >
          (std::numeric_limits<std::uint64_t>::max() - start.value) / 4)
    return std::unexpected(error{error_code::block_too_large, 0});
  let created = ir::block::create(
      {start.value, static_cast<std::uint32_t>(words.size()), 32},
      words.size() * nodes_per_instruction + extra_nodes, memory);
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
    const bool hilo_read =
        opcode == 0 && (function == 0x10 || function == 0x12);
    const bool hilo_write =
        opcode == 0 && (function == 0x11 || function == 0x13);
    const bool syscall = opcode == 0 && function == 0x0c;
    const bool word_shift =
        opcode == 0 && (function == 0 || function == 2 || function == 3);
    const bool addu = opcode == 0 && function == 0x21;
    const bool mfc0 = opcode == 0x10 && (rs == 0 || rs == 1);
    const bool mtc0 = opcode == 0x10 && (rs == 4 || rs == 5);
    const bool lui = opcode == 0x0f;
    const bool sltiu = opcode == 0x0b;
    const bool slti = opcode == 0x0a;
    const bool daddiu = opcode == 0x19;
    const bool addiu = opcode == 0x09;
    const bool immediate = opcode == 0x0c || opcode == 0x0d || lui || addiu ||
                           daddiu || slti || sltiu;
    const bool load = opcode == 0x20 || opcode == 0x23 || opcode == 0x37;
    const bool store = opcode == 0x28 || opcode == 0x2b || opcode == 0x3f;
    const bool memory_access = load || store;
    const bool branch = is_branch(words[i]);
    const bool jump = opcode == 2 || opcode == 3;
    const bool indirect = opcode == 0 && (function == 8 || function == 9);
    if (!immediate && !memory_access && !branch && !jump && !indirect &&
        !mtc0 && !mfc0 && !word_shift && !syscall && !hilo_read &&
        !hilo_write && !addu &&
        !(opcode == 0 &&
          (function == 0x24 || function == 0x26 || function == 0x2d)))
      return std::unexpected(error{error_code::unsupported_instruction, i});
    if (!immediate && !memory_access && !branch && !jump && !indirect &&
        !mtc0 && !mfc0 && !word_shift && !syscall && !hilo_read &&
        !hilo_write && ((bits >> 6) & 31) != 0)
      return std::unexpected(error{error_code::invalid_encoding, i});
    if ((hilo_read && (rs != 0 || rt != 0 || ((bits >> 6) & 31) != 0)) ||
        (hilo_write && (rt != 0 || rd != 0 || ((bits >> 6) & 31) != 0)))
      return std::unexpected(error{error_code::invalid_encoding, i});
    if (word_shift && rs != 0)
      return std::unexpected(error{error_code::invalid_encoding, i});
    if ((mtc0 || mfc0) &&
        ((bits & 0x7ff) != 0 ||
         (rd != 9 && rd != 11 && rd != 12 && rd != 13 && rd != 15 && rd != 16 &&
          rd != 8 && rd != 14 && rd != 30)))
      return std::unexpected(error{error_code::unsupported_instruction, i});
    if (indirect && (rt != 0 || (function == 8 && rd != 0) ||
                     (function == 9 && rd == rs) || ((bits >> 6) & 31) != 0))
      return std::unexpected(error{error_code::invalid_encoding, i});
    if (lui && rs != 0)
      return std::unexpected(error{error_code::invalid_encoding, i});
    const let destination = mfc0 || immediate || memory_access ? rt : rd;
    if (destination == 0 && !memory_access && !branch && !jump && !indirect &&
        !mtc0 && !syscall && !hilo_write)
      continue;
    const let source = static_cast<std::uint32_t>(i);
    const let failed = [&](ir::error failure) {
      return std::unexpected(error{failure.code == ir::error_code::out_of_memory
                                       ? error_code::out_of_memory
                                       : error_code::invalid_ir,
                                   i});
    };
    if (syscall) {
      if (let raised = result.append(ir::node::syscall(source)); !raised)
        return failed(raised.error());
      continue;
    }
    if (lui) {
      const let bits32 = static_cast<std::int32_t>((bits & 0xffffu) << 16);
      const let value = result.append(ir::node::constant(
          static_cast<std::uint64_t>(std::int64_t{bits32}), source));
      if (!value)
        return failed(value.error());
      if (let written = result.append(ir::node::write({rt}, *value, source));
          !written)
        return failed(written.error());
      registers[rt] = *value;
      continue;
    }
    if (jump) {
      const guest_address pc{start.value + i * 4};
      if (opcode == 3) {
        const let link =
            result.append(ir::node::constant(pc.value + 8, source));
        if (!link)
          return failed(link.error());
        if (let written = result.append(ir::node::write({31}, *link, source));
            !written)
          return failed(written.error());
        registers[31] = *link;
      }
      if (let exit =
              result.set_jump({jump_target(words[i], pc).value, source});
          !exit)
        return failed(exit.error());
      continue;
    }
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
    if (hilo_read || hilo_write) {
      const ir::state_slot slot{
          static_cast<std::uint16_t>((function >> 1) & 1)};
      if (hilo_read) {
        const let value = result.append(ir::node::read_hilo(slot, source));
        if (!value)
          return failed(value.error());
        if (let written = result.append(ir::node::write({rd}, *value, source));
            !written)
          return failed(written.error());
        registers[rd] = *value;
      } else {
        const let value = read(rs);
        if (!value)
          return failed(value.error());
        if (let written =
                result.append(ir::node::write_hilo(slot, *value, source));
            !written)
          return failed(written.error());
      }
      continue;
    }
    if (word_shift) {
      const let input = read(rt);
      if (!input)
        return failed(input.error());
      const let value = result.append(
          function == 0
              ? ir::node::shift_left32(*input, (bits >> 6) & 31, source)
          : function == 2
              ? ir::node::shift_right32(*input, (bits >> 6) & 31, source)
              : ir::node::shift_right_arithmetic32(*input, (bits >> 6) & 31,
                                                   source));
      if (!value)
        return failed(value.error());
      if (let written = result.append(ir::node::write({rd}, *value, source));
          !written)
        return failed(written.error());
      registers[rd] = *value;
      continue;
    }
    if (mfc0) {
      const let value =
          result.append(rs == 1 ? ir::node::read_cop0_64({rd}, source)
                                : ir::node::read_cop0({rd}, source));
      if (!value)
        return failed(value.error());
      if (let written = result.append(ir::node::write({rt}, *value, source));
          !written)
        return failed(written.error());
      registers[rt] = *value;
      continue;
    }
    if (mtc0) {
      const let value = read(rt);
      if (!value)
        return failed(value.error());
      if (let written = result.append(
              rs == 5 ? ir::node::write_cop0_64({rd}, *value, source)
                      : ir::node::write_cop0({rd}, *value, source));
          !written)
        return failed(written.error());
      continue;
    }
    const let left = read(rs);
    if (!left)
      return failed(left.error());
    if (indirect) {
      if (function == 9 && rd != 0) {
        const let link =
            result.append(ir::node::constant(start.value + i * 4 + 8, source));
        if (!link)
          return failed(link.error());
        if (let written = result.append(ir::node::write({rd}, *link, source));
            !written)
          return failed(written.error());
        registers[rd] = *link;
      }
      if (let exit = result.set_indirect({*left, source}); !exit)
        return failed(exit.error());
      continue;
    }
    if (branch) {
      const bool sign_branch = opcode == 1;
      const let right = read(sign_branch ? 0 : rt);
      if (!right)
        return failed(right.error());
      let condition = *left;
      if (sign_branch) {
        const let negative = result.append(ir::node::binary(
            ir::binary_kind::signed_less, *left, *right, source));
        if (!negative)
          return failed(negative.error());
        condition = *negative;
      }
      const let displacement =
          std::int64_t{static_cast<std::int16_t>(bits & 0xffff)} * 4;
      const let target =
          start.value + i * 4 + 4 + static_cast<std::uint64_t>(displacement);
      const bool equal =
          sign_branch ? (rt & 1) != 0 : opcode == 4 || opcode == 0x14;
      if (let exit = result.set_branch(
              {condition, *right, target, source,
               equal ? ir::branch_kind::equal : ir::branch_kind::not_equal,
               sign_branch ? rt >= 2 : opcode == 0x14 || opcode == 0x15});
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
        if (let written = result.append(
                opcode == 0x28
                    ? ir::node::store8(*address, *value, source)
                    : ir::node::store(*address, *value, source, size));
            !written)
          return failed(written.error());
      } else {
        const let value = result.append(
            opcode == 0x20 ? ir::node::load8_signed(*address, source)
                           : ir::node::load(*address, source, size));
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
        immediate ? result.append(ir::node::constant(
                        (addiu || daddiu || slti || sltiu)
                            ? static_cast<std::uint64_t>(std::int64_t{
                                  static_cast<std::int16_t>(bits & 0xffff)})
                            : bits & 0xffff,
                        source))
                  : read(rt);
    if (!right)
      return failed(right.error());
    const let operation = sltiu               ? ir::binary_kind::unsigned_less
                          : slti              ? ir::binary_kind::signed_less
                          : opcode == 0x0c    ? ir::binary_kind::bit_and
                          : (addiu || daddiu) ? ir::binary_kind::add_wrap
                          : immediate         ? ir::binary_kind::bit_or
                          : function == 0x24  ? ir::binary_kind::bit_and
                          : function == 0x26 ? ir::binary_kind::bit_xor
                                             : ir::binary_kind::add_wrap;
    let value =
        result.append(ir::node::binary(operation, *left, *right, source));
    if (!value)
      return failed(value.error());
    if (addiu || addu) {
      value = result.append(ir::node::sign_extend32(*value, source));
      if (!value)
        return failed(value.error());
    }
    if (let written =
            result.append(ir::node::write({destination}, *value, source));
        !written)
      return failed(written.error());
    registers[destination] = *value;
  }
  return result;
}

} // namespace jit::mips64
