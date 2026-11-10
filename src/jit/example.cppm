module;

#include <array>
#include <cstdint>

#include <dynamips/macros.h>

export module jit.example;

import jit.mips64;

namespace jit::example {
enum class reg : std::uint32_t {
  zero,
  console,
  value,
  status,
  ram,
  left,
  right,
  sum,
  readback,
  pointer,
  check,
  ascii_zero
};

constexpr fn immediate(std::uint32_t opcode, reg target, reg base,
                        std::uint16_t value) noexcept
    -> mips64::instruction_word {
  return {(opcode << 26) | (static_cast<std::uint32_t>(base) << 21) |
          (static_cast<std::uint32_t>(target) << 16) | value};
}

constexpr fn ori(reg target, std::uint16_t value) noexcept
    -> mips64::instruction_word {
  return immediate(0x0d, target, reg::zero, value);
}

constexpr fn lw(reg target, reg base, std::int16_t offset) noexcept
    -> mips64::instruction_word {
  return immediate(0x23, target, base, static_cast<std::uint16_t>(offset));
}

constexpr fn sw(reg value, reg base, std::uint16_t offset) noexcept
    -> mips64::instruction_word {
  return immediate(0x2b, value, base, offset);
}

constexpr fn ld(reg target, reg base, std::int16_t offset) noexcept
    -> mips64::instruction_word {
  return immediate(0x37, target, base, static_cast<std::uint16_t>(offset));
}

constexpr fn sd(reg value, reg base, std::uint16_t offset) noexcept
    -> mips64::instruction_word {
  return immediate(0x3f, value, base, offset);
}

constexpr fn binary(std::uint32_t function, reg target, reg left,
                     reg right) noexcept -> mips64::instruction_word {
  return {(static_cast<std::uint32_t>(left) << 21) |
          (static_cast<std::uint32_t>(right) << 16) |
          (static_cast<std::uint32_t>(target) << 11) | function};
}

constexpr fn daddu(reg target, reg left, reg right) noexcept
    -> mips64::instruction_word {
  return binary(0x2d, target, left, right);
}

constexpr fn bit_xor(reg target, reg left, reg right) noexcept
    -> mips64::instruction_word {
  return binary(0x26, target, left, right);
}

export inline constexpr std::array program{
    ori(reg::console, 0x8000),
    ori(reg::ram, 0x2000),
    ori(reg::left, 4),
    ori(reg::right, 2),
    sd(reg::left, reg::ram, 0),
    sd(reg::right, reg::ram, 8),
    ld(reg::left, reg::ram, 0),
    ld(reg::right, reg::ram, 8),
    daddu(reg::sum, reg::left, reg::right),
    sd(reg::sum, reg::ram, 16),
    ori(reg::pointer, 0x2018),
    ld(reg::readback, reg::pointer, -8),
    bit_xor(reg::check, reg::sum, reg::readback),
    sw(reg::check, reg::ram, 24),
    ori(reg::ascii_zero, '0'),
    daddu(reg::value, reg::readback, reg::ascii_zero),
    sw(reg::value, reg::ram, 80),
    ori(reg::value, '\n'),
    sw(reg::value, reg::ram, 84),
    ori(reg::value, 's'),
    sw(reg::value, reg::ram, 64),
    ori(reg::value, 'u'),
    sw(reg::value, reg::ram, 68),
    ori(reg::value, 'm'),
    sw(reg::value, reg::ram, 72),
    ori(reg::value, '='),
    sw(reg::value, reg::ram, 76),
    lw(reg::value, reg::ram, 64),
    lw(reg::status, reg::console, 4),
    sw(reg::value, reg::console, 0),
    lw(reg::value, reg::ram, 68),
    lw(reg::status, reg::console, 4),
    sw(reg::value, reg::console, 0),
    lw(reg::value, reg::ram, 72),
    lw(reg::status, reg::console, 4),
    sw(reg::value, reg::console, 0),
    lw(reg::value, reg::ram, 76),
    lw(reg::status, reg::console, 4),
    sw(reg::value, reg::console, 0),
    lw(reg::value, reg::ram, 80),
    lw(reg::status, reg::console, 4),
    sw(reg::value, reg::console, 0),
    lw(reg::value, reg::ram, 84),
    lw(reg::status, reg::console, 4),
    sw(reg::value, reg::console, 0),
    lw(reg::status, reg::console, 4),
    sw(reg::status, reg::ram, 28),
    ld(reg::readback, reg::ram, 16)};
} // namespace jit::example
