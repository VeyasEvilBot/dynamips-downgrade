module;

#include <array>
#include <bit>
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
import jit.tlb;

export namespace jit::mips64 {

struct instruction_word {
  std::uint32_t bits;
};

struct guest_address {
  std::uint64_t value;
};

constexpr fn is_branch(instruction_word word) noexcept -> bool {
  const let op = word.bits >> 26;
  return op == 6 || op == 7 || op == 4 || op == 5 || op == 0x14 || op == 0x15 ||
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
  tlb::table translations;
  std::array<std::uint64_t, 32> registers{};
  bool half_cycle = false;

  // Count ticks every two virtual CPU cycles. Dispatch uses instruction slots
  // as a deterministic cycle estimate, observed at block boundaries.
  fn advance_cycles(std::uint64_t cycles) noexcept -> void {
    const let odd = (cycles & 1) != 0;
    const let ticks = cycles / 2 + (odd && half_cycle ? 1u : 0u);
    half_cycle = half_cycle != odd;
    const let before = static_cast<std::uint32_t>(registers[9]);
    const let distance = static_cast<std::uint32_t>(registers[11] - before);
    registers[9] = static_cast<std::uint32_t>(before + ticks);
    if ((distance != 0 && distance <= ticks) ||
        ticks >= (std::uint64_t{1} << 32))
      registers[13] |= 1u << 15;
  }

  fn interrupt_pending() const noexcept -> bool {
    return (registers[12] & 7) == 1 &&
           (registers[12] & registers[13] & 0xff00) != 0;
  }

  fn exception(std::uint32_t code, guest_address pc, bool delay_slot) noexcept
      -> guest_address {
    let &status = registers[12];
    let &cause = registers[13];
    cause = (cause & ~std::uint64_t{0x7c}) | (std::uint64_t{code} << 2);
    if (!(status & 2)) {
      registers[14] = pc.value;
      cause = (cause & ~(std::uint64_t{1} << 31)) |
              (delay_slot ? std::uint64_t{1} << 31 : 0);
    }
    status |= 2;
    return {status & (1u << 22) ? 0xffffffffbfc00380ULL
                                : 0xffffffff80000180ULL};
  }

  fn exception_return() noexcept -> guest_address {
    let &status = registers[12];
    const bool error_level = (status & 4) != 0;
    status &= ~std::uint64_t{error_level ? 4u : 2u};
    return {registers[error_level ? 30 : 14]};
  }

  fn write(std::uint16_t slot, std::uint32_t value) noexcept -> void {
    if (slot == 7 || slot == 8 || slot == 15 || slot == 16)
      return;
    if (slot == 0 || slot == 6)
      value &= 63;
    else if (slot == 5)
      value &= 0x01ffe000;
    else if (slot == 2 || slot == 3)
      value &= 0x7fffffff;
    else if (slot == 10)
      value &= 0xffffe0ff;
    registers[slot] = value;
    if (slot == 11)
      registers[13] &= ~(1u << 15);
  }
};

struct hilo_state {
  std::array<std::uint64_t, 2> registers{};

  fn multiply32(std::uint64_t left, std::uint64_t right) noexcept -> void {
    const let product =
        std::int64_t{
            std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(left))} *
        std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(right));
    const let bits = static_cast<std::uint64_t>(product);
    write_words(static_cast<std::uint32_t>(bits >> 32),
                static_cast<std::uint32_t>(bits));
  }

  fn multiply_unsigned32(std::uint64_t left, std::uint64_t right) noexcept
      -> void {
    const let product = std::uint64_t{static_cast<std::uint32_t>(left)} *
                        static_cast<std::uint32_t>(right);
    write_words(static_cast<std::uint32_t>(product >> 32),
                static_cast<std::uint32_t>(product));
  }

  fn divide_signed32(std::uint64_t left, std::uint64_t right) noexcept
      -> void {
    const std::int64_t numerator =
        std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(left));
    const std::int64_t denominator =
        std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(right));
    if (denominator == 0)
      return;
    write_words(static_cast<std::uint32_t>(numerator % denominator),
                static_cast<std::uint32_t>(numerator / denominator));
  }

  fn divide_unsigned32(std::uint64_t left, std::uint64_t right) noexcept
      -> void {
    const let numerator = static_cast<std::uint32_t>(left);
    const let denominator = static_cast<std::uint32_t>(right);
    // Preserve HI/LO for the unpredictable zero-divisor case, as in legacy
    // DIVU.
    if (denominator == 0)
      return;
    write_words(numerator % denominator, numerator / denominator);
  }

private:
  fn write_words(std::uint32_t high, std::uint32_t low) noexcept -> void {
    registers[0] = static_cast<std::uint64_t>(
        std::int64_t{std::bit_cast<std::int32_t>(high)});
    registers[1] = static_cast<std::uint64_t>(
        std::int64_t{std::bit_cast<std::int32_t>(low)});
  }
};

struct cop1_state {
  std::array<std::uint64_t, 32> registers{};
};

struct state {
  std::array<std::uint64_t, 32> registers{};
  guest_address pc{};
  cop0_state cop0{};
  hilo_state hilo{};
  cop1_state cop1{};

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
    const let count_read = (words[i].bits >> 26) == 0x10 &&
                           ((words[i].bits >> 21) & 31) <= 1 &&
                           ((words[i].bits >> 11) & 31) == 9;
    if (count_read || words[i].bits == 0x42000001 ||
        words[i].bits == 0x42000002 || words[i].bits == 0x42000008 ||
        words[i].bits >> 26 == 0x2f || words[i].bits == 0x42000018 ||
        ((words[i].bits >> 26) == 0 &&
         ((words[i].bits & 63) == 0x0c || (words[i].bits & 63) == 0x0f))) {
      words = words.first(i + 1);
      break;
    }
    if (!has_delay_slot(words[i]))
      continue;
    if (i + 1 == words.size())
      return std::unexpected(error{error_code::missing_delay_slot, i});
    if (words[i + 1].bits == 0x42000018)
      return std::unexpected(error{error_code::invalid_encoding, i + 1});
    if (has_delay_slot(words[i + 1]))
      return std::unexpected(error{error_code::branch_in_delay_slot, i + 1});
    words = words.first(i + 2);
    break;
  }
  constexpr std::size_t nodes_per_instruction = 4;
  const let extra_nodes =
      static_cast<std::size_t>(std::ranges::count_if(
          words,
          [](instruction_word word) {
            const let op = word.bits >> 26;
            return (op == 0 &&
                    ((word.bits & 63) == 0x21 || (word.bits & 63) == 0x23)) ||
                   op == 0x21 || op == 0x25 || op == 0x29 || op == 0x09 ||
                   op == 0x20 || op == 0x24 || op == 0x23 || op == 0x28 ||
                   op == 0x2b || op == 0x2a || op == 0x2e || op == 0x2c ||
                   op == 0x2d || op == 0x37 || op == 0x3f;
          })) +
      2 * static_cast<std::size_t>(
              std::ranges::count_if(words, [](instruction_word word) {
                return (word.bits >> 26 == 0 && (word.bits & 63) == 0x27) ||
                       word.bits >> 26 == 0x1a || word.bits >> 26 == 0x1b ||
                       word.bits >> 26 == 0x22 || word.bits >> 26 == 0x26;
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
    // PREF is advisory; coherent memory needs no prefetch transaction.
    if (opcode == 0x33)
      continue;
    const let rs = static_cast<std::uint16_t>((bits >> 21) & 31);
    const let rt = static_cast<std::uint16_t>((bits >> 16) & 31);
    const let rd = static_cast<std::uint16_t>((bits >> 11) & 31);
    const let function = bits & 63;
    const bool hilo_arithmetic =
        opcode == 0 && (function == 0x18 || function == 0x19 ||
                        function == 0x1a || function == 0x1b);
    const bool hilo_read =
        opcode == 0 && (function == 0x10 || function == 0x12);
    const bool hilo_write =
        opcode == 0 && (function == 0x11 || function == 0x13);
    const bool cache = opcode == 0x2f;
    const bool tlb_operation =
        bits == 0x42000001 || bits == 0x42000002 || bits == 0x42000008;
    const bool eret = bits == 0x42000018;
    const bool sync = opcode == 0 && function == 0x0f;
    const bool syscall = opcode == 0 && function == 0x0c;
    const bool variable_shift =
        opcode == 0 && (function == 4 || function == 6 || function == 7);
    const bool word_shift =
        opcode == 0 && (function == 0 || function == 2 || function == 3);
    const bool double_shift =
        opcode == 0 &&
        (function == 0x38 || function == 0x3a || function == 0x3b ||
         function == 0x3c || function == 0x3e || function == 0x3f);
    const bool nor = opcode == 0 && function == 0x27;
    const bool subu = opcode == 0 && function == 0x23;
    const bool slt = opcode == 0 && function == 0x2a;
    const bool sltu = opcode == 0 && function == 0x2b;
    const bool addu = opcode == 0 && function == 0x21;
    const bool dmfc1 = opcode == 0x11 && rs == 1;
    const bool dmtc1 = opcode == 0x11 && rs == 5;
    const bool mfc0 = opcode == 0x10 && (rs == 0 || rs == 1);
    const bool mtc0 = opcode == 0x10 && (rs == 4 || rs == 5);
    const bool lui = opcode == 0x0f;
    const bool sltiu = opcode == 0x0b;
    const bool slti = opcode == 0x0a;
    const bool daddiu = opcode == 0x19;
    const bool addi = opcode == 0x08;
    const bool addiu = opcode == 0x09;
    const bool immediate = opcode == 0x0c || opcode == 0x0d || opcode == 0x0e ||
                           lui || addi || addiu || daddiu || slti || sltiu;
    const bool merge_load =
        opcode == 0x1a || opcode == 0x1b || opcode == 0x22 || opcode == 0x26;
    const bool load = merge_load || opcode == 0x21 || opcode == 0x25 ||
                      opcode == 0x20 || opcode == 0x24 || opcode == 0x23 ||
                      opcode == 0x37;
    const bool merge_store =
        opcode == 0x2a || opcode == 0x2e || opcode == 0x2c || opcode == 0x2d;
    const bool store = merge_store || opcode == 0x29 || opcode == 0x28 ||
                       opcode == 0x2b || opcode == 0x3f;
    const bool memory_access = load || store;
    const bool branch = is_branch(words[i]);
    const bool jump = opcode == 2 || opcode == 3;
    const bool indirect = opcode == 0 && (function == 8 || function == 9);
    if (!immediate && !memory_access && !branch && !jump && !indirect &&
        !mtc0 && !mfc0 && !dmfc1 && !dmtc1 && !word_shift && !double_shift &&
        !syscall && !sync && !tlb_operation && !eret && !cache && !hilo_read &&
        !hilo_write && !addu && !subu && !slt && !sltu && !variable_shift &&
        !nor && !hilo_arithmetic &&
        !(opcode == 0 && (function == 0x24 || function == 0x25 ||
                          function == 0x26 || function == 0x2d)))
      return std::unexpected(error{error_code::unsupported_instruction, i});
    if (!immediate && !memory_access && !branch && !jump && !indirect &&
        !mtc0 && !mfc0 && !dmfc1 && !dmtc1 && !word_shift && !double_shift &&
        !syscall && !sync && !tlb_operation && !eret && !cache && !hilo_read &&
        !hilo_write && ((bits >> 6) & 31) != 0)
      return std::unexpected(error{error_code::invalid_encoding, i});
    if ((hilo_read && (rs != 0 || rt != 0 || ((bits >> 6) & 31) != 0)) ||
        (hilo_write && (rt != 0 || rd != 0 || ((bits >> 6) & 31) != 0)))
      return std::unexpected(error{error_code::invalid_encoding, i});
    if (sync && (bits & 0xfffff83fu) != 0x0000000fu)
      return std::unexpected(error{error_code::invalid_encoding, i});
    if (cache && rt > 1 && rt != 11 && rt != 22)
      return std::unexpected(error{error_code::unsupported_instruction, i});
    if (hilo_arithmetic && (rd != 0 || ((bits >> 6) & 31) != 0))
      return std::unexpected(error{error_code::invalid_encoding, i});
    if ((word_shift || double_shift) && rs != 0)
      return std::unexpected(error{error_code::invalid_encoding, i});
    if ((dmfc1 || dmtc1) && (bits & 0x7ff) != 0)
      return std::unexpected(error{error_code::invalid_encoding, i});
    if ((mtc0 || mfc0) &&
        ((bits & 0x7ff) != 0 ||
         (rd != 9 && rd != 11 && rd != 12 && rd != 13 && rd != 15 && rd != 16 &&
          rd != 7 && rd != 8 && rd != 14 && rd != 30 && rd != 0 && rd != 2 &&
          rd != 3 && rd != 5 && rd != 6 && rd != 10 && rd != 28 && rd != 29)))
      return std::unexpected(error{error_code::unsupported_instruction, i});
    if (indirect && (rt != 0 || (function == 8 && rd != 0) ||
                     (function == 9 && rd == rs) || ((bits >> 6) & 31) != 0))
      return std::unexpected(error{error_code::invalid_encoding, i});
    if ((opcode == 6 || opcode == 7) && rt != 0)
      return std::unexpected(error{error_code::invalid_encoding, i});
    if (lui && rs != 0)
      return std::unexpected(error{error_code::invalid_encoding, i});
    const let destination =
        dmfc1 || mfc0 || immediate || memory_access ? rt : rd;
    if (destination == 0 && !memory_access && !branch && !jump && !indirect &&
        !mtc0 && !dmtc1 && !syscall && !sync && !tlb_operation && !eret &&
        !cache && !hilo_write && !addi && !hilo_arithmetic)
      continue;
    const let source = static_cast<std::uint32_t>(i);
    const let failed = [&](ir::error failure) {
      return std::unexpected(error{failure.code == ir::error_code::out_of_memory
                                       ? error_code::out_of_memory
                                       : error_code::invalid_ir,
                                   i});
    };
    if (tlb_operation) {
      if (let operated = result.append(ir::node::tlb_operation(
              bits == 0x42000001   ? ir::tlb_action::read
              : bits == 0x42000002 ? ir::tlb_action::write_indexed
                                   : ir::tlb_action::probe,
              source));
          !operated)
        return failed(operated.error());
      continue;
    }
    if (sync) {
      if (let fenced = result.append(ir::node::memory_barrier(source));
          !fenced)
        return failed(fenced.error());
      continue;
    }
    if (eret) {
      if (let returned = result.append(ir::node::exception_return(source));
          !returned)
        return failed(returned.error());
      continue;
    }
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
    if (cache) {
      const let base = read(rs);
      if (!base)
        return failed(base.error());
      const let offset = result.append(
          ir::node::constant(static_cast<std::uint64_t>(std::int64_t{
                                 static_cast<std::int16_t>(bits & 0xffff)}),
                             source));
      if (!offset)
        return failed(offset.error());
      const let address = result.append(
          ir::node::binary(ir::binary_kind::add_wrap, *base, *offset, source));
      if (!address)
        return failed(address.error());
      if (let maintained =
              result.append(ir::node::cache_index(rt, *address, source));
          !maintained)
        return failed(maintained.error());
      continue;
    }
    if (hilo_arithmetic) {
      const let left = read(rs);
      const let right = read(rt);
      if (!left || !right)
        return failed(!left ? left.error() : right.error());
      if (let written = result.append(
              function == 0x18 ? ir::node::multiply32(*left, *right, source)
              : function == 0x19
                  ? ir::node::multiply_unsigned32(*left, *right, source)
              : function == 0x1a
                  ? ir::node::divide_signed32(*left, *right, source)
                  : ir::node::divide_unsigned32(*left, *right, source));
          !written)
        return failed(written.error());
      continue;
    }
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
    if (variable_shift) {
      const let input = read(rt);
      const let count = read(rs);
      if (!input || !count)
        return failed(!input ? input.error() : count.error());
      const let operation =
          function == 4   ? ir::binary_kind::shift_left32_variable
          : function == 6 ? ir::binary_kind::shift_right32_variable
                          : ir::binary_kind::shift_right_arithmetic32_variable;
      const let value =
          result.append(ir::node::binary(operation, *input, *count, source));
      if (!value)
        return failed(value.error());
      if (let written = result.append(ir::node::write({rd}, *value, source));
          !written)
        return failed(written.error());
      registers[rd] = *value;
      continue;
    }
    if (double_shift) {
      const let input = read(rt);
      if (!input)
        return failed(input.error());
      const let amount = ((bits >> 6) & 31) + ((function & 4) ? 32 : 0);
      const let value = result.append(
          (function & 3) == 0 ? ir::node::shift_left64(*input, amount, source)
          : (function & 3) == 2
              ? ir::node::shift_right64(*input, amount, source)
              : ir::node::shift_right_arithmetic64(*input, amount, source));
      if (!value)
        return failed(value.error());
      if (let written = result.append(ir::node::write({rd}, *value, source));
          !written)
        return failed(written.error());
      registers[rd] = *value;
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
    if (dmfc1 || dmtc1) {
      if (dmfc1) {
        const let value = result.append(ir::node::read_cop1_64({rd}, source));
        if (!value)
          return failed(value.error());
        if (let written = result.append(ir::node::write({rt}, *value, source));
            !written)
          return failed(written.error());
        registers[rt] = *value;
      } else {
        const let value = read(rt);
        if (!value)
          return failed(value.error());
        if (let written =
                result.append(ir::node::write_cop1_64({rd}, *value, source));
            !written)
          return failed(written.error());
      }
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
      const bool sign_branch = opcode == 1 || opcode == 6 || opcode == 7;
      const bool positive = opcode == 6 || opcode == 7;
      const let right = read(sign_branch ? 0 : rt);
      if (!right)
        return failed(right.error());
      let condition = *left;
      if (sign_branch) {
        const let negative = result.append(ir::node::binary(
            ir::binary_kind::signed_less, positive ? *right : *left,
            positive ? *left : *right, source));
        if (!negative)
          return failed(negative.error());
        condition = *negative;
      }
      const let displacement =
          std::int64_t{static_cast<std::int16_t>(bits & 0xffff)} * 4;
      const let target =
          start.value + i * 4 + 4 + static_cast<std::uint64_t>(displacement);
      const bool equal = positive      ? opcode == 6
                         : sign_branch ? (rt & 1) != 0
                                       : opcode == 4 || opcode == 0x14;
      if (let exit = result.set_branch(
              {condition, *right, target, source,
               equal ? ir::branch_kind::equal : ir::branch_kind::not_equal,
               opcode == 1 ? rt >= 2 : opcode == 0x14 || opcode == 0x15});
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
      if (merge_load) {
        const let previous = read(rt);
        if (!previous)
          return failed(previous.error());
        const let value =
            result.append((opcode == 0x22 || opcode == 0x26)
                              ? ir::node::load_merge32(*address, *previous,
                                                       opcode == 0x26, source)
                              : ir::node::load_merge64(*address, *previous,
                                                       opcode == 0x1b, source));
        if (!value)
          return failed(value.error());
        if (rt != 0) {
          if (let written =
                  result.append(ir::node::write({rt}, *value, source));
              !written)
            return failed(written.error());
          registers[rt] = *value;
        }
        continue;
      }
      const let size = opcode == 0x23 || opcode == 0x2b ? ir::width::bits32
                                                        : ir::width::bits64;
      if (store) {
        const let value = read(rt);
        if (!value)
          return failed(value.error());
        if (let written = result.append(
                merge_store
                    ? ((opcode == 0x2c || opcode == 0x2d)
                           ? ir::node::store_merge64(*address, *value,
                                                     opcode == 0x2d, source)
                           : ir::node::store_merge32(*address, *value,
                                                     opcode == 0x2e, source))
                : opcode == 0x29 ? ir::node::store16(*address, *value, source)
                : opcode == 0x28
                    ? ir::node::store8(*address, *value, source)
                    : ir::node::store(*address, *value, source, size));
            !written)
          return failed(written.error());
      } else {
        const let value = result.append(
            opcode == 0x21   ? ir::node::load16_signed(*address, source)
            : opcode == 0x25 ? ir::node::load16_unsigned(*address, source)
            : opcode == 0x24 ? ir::node::load8_unsigned(*address, source)
            : opcode == 0x20 ? ir::node::load8_signed(*address, source)
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
                        (addi || addiu || daddiu || slti || sltiu)
                            ? static_cast<std::uint64_t>(std::int64_t{
                                  static_cast<std::int16_t>(bits & 0xffff)})
                            : bits & 0xffff,
                        source))
                  : read(rt);
    if (!right)
      return failed(right.error());
    const let operation = subu                ? ir::binary_kind::subtract_wrap
                          : (sltiu || sltu)   ? ir::binary_kind::unsigned_less
                          : (slti || slt)     ? ir::binary_kind::signed_less
                          : opcode == 0x0e    ? ir::binary_kind::bit_xor
                          : opcode == 0x0c    ? ir::binary_kind::bit_and
                          : (addiu || daddiu) ? ir::binary_kind::add_wrap
                          : immediate         ? ir::binary_kind::bit_or
                          : function == 0x24  ? ir::binary_kind::bit_and
                          : (function == 0x25 || nor) ? ir::binary_kind::bit_or
                          : function == 0x26 ? ir::binary_kind::bit_xor
                                             : ir::binary_kind::add_wrap;
    let value = result.append(
        addi ? ir::node::add_checked32(*left, *right, source)
             : ir::node::binary(operation, *left, *right, source));
    if (!value)
      return failed(value.error());
    if (nor) {
      const let mask = result.append(ir::node::constant(UINT64_MAX, source));
      if (!mask)
        return failed(mask.error());
      value = result.append(
          ir::node::binary(ir::binary_kind::bit_xor, *value, *mask, source));
      if (!value)
        return failed(value.error());
    }
    if (addi && destination == 0)
      continue;
    if (addiu || addu || subu) {
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
