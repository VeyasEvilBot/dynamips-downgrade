module;

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>

#include <dynamips/macros.h>

export module jit.interpreter;

import jit.memory;
import jit.mips64;

export namespace jit::interpreter {
struct result {
  std::uint64_t next_pc;
  std::uint32_t instructions;
  std::uint32_t accounted_instructions = 0;
};

enum class error_code { memory, unsupported };

struct failure {
  memory::fault access{};
  std::uint64_t pc = 0;
  std::optional<std::uint64_t> branch_pc{};
  std::uint32_t completed_instructions = 0;
  error_code code = error_code::memory;
  std::uint32_t instructions = 0;
};

namespace detail {
constexpr fn control(mips64::instruction_word word) noexcept -> bool {
  const let op = word.bits >> 26;
  const let function = word.bits & 63;
  return op == 1 || (op >= 2 && op <= 7) || op == 0x14 || op == 0x15 ||
         (op == 0 && (function == 8 || function == 9));
}

constexpr fn supported(mips64::instruction_word word) noexcept -> bool {
  const let op = word.bits >> 26;
  const let rs = (word.bits >> 21) & 31;
  const let rt = (word.bits >> 16) & 31;
  const let rd = (word.bits >> 11) & 31;
  const let shift = (word.bits >> 6) & 31;
  if (op == 0) {
    switch (word.bits & 63) {
    case 0:
    case 2:
    case 3:
    case 0x38:
    case 0x3a:
    case 0x3b:
    case 0x3c:
    case 0x3e:
    case 0x3f:
      return rs == 0;
    case 4:
    case 6:
    case 7:
    case 0x21:
    case 0x23:
    case 0x24:
    case 0x25:
    case 0x26:
    case 0x27:
    case 0x2a:
    case 0x2b:
    case 0x2d:
      return shift == 0;
    case 8:
      return rt == 0 && rd == 0 && shift == 0;
    case 9:
      return rt == 0 && rd != rs && shift == 0;
    default:
      return false;
    }
  }
  switch (op) {
  case 1:
    return rt <= 3;
  case 6:
  case 7:
    return rt == 0;
  case 0x0f:
    return rs == 0;
  case 2:
  case 3:
  case 4:
  case 5:
  case 9:
  case 0x0a:
  case 0x0b:
  case 0x0c:
  case 0x0d:
  case 0x0e:
  case 0x14:
  case 0x15:
  case 0x19:
  case 0x1a:
  case 0x1b:
  case 0x20:
  case 0x21:
  case 0x22:
  case 0x23:
  case 0x24:
  case 0x25:
  case 0x26:
  case 0x27:
  case 0x28:
  case 0x29:
  case 0x2a:
  case 0x2b:
  case 0x2c:
  case 0x2d:
  case 0x2e:
  case 0x30:
  case 0x37:
  case 0x38:
  case 0x3f:
    return true;
  default:
    return false;
  }
}

constexpr fn signed_word(std::uint32_t value) noexcept -> std::uint64_t {
  return static_cast<std::uint64_t>(
      std::int64_t{std::bit_cast<std::int32_t>(value)});
}

struct execution {
  mips64::cop0_state &clock;
  std::uint32_t completed = 0;
  std::uint32_t accounted = 0;

  static fn before_external(void *opaque) noexcept -> void {
    let &self = *static_cast<execution *>(opaque);
    self.clock.advance_cycles(self.completed - self.accounted);
    self.accounted = self.completed;
  }
};

template <typename T, bool Signed = false>
fn load(memory::address_space &space, std::uint64_t address) noexcept
    -> std::expected<std::uint64_t, memory::fault> {
  const let value = space.load<T>({address});
  if (!value)
    return std::unexpected(value.error());
  if constexpr (Signed && sizeof(T) == 1)
    return static_cast<std::uint64_t>(
        std::int64_t{std::bit_cast<std::int8_t>(*value)});
  else if constexpr (Signed && sizeof(T) == 2)
    return static_cast<std::uint64_t>(
        std::int64_t{std::bit_cast<std::int16_t>(*value)});
  else if constexpr (Signed && sizeof(T) == 4)
    return signed_word(*value);
  else
    return *value;
}

template <typename T, bool Right>
fn load_merge(memory::address_space &space, std::uint64_t address,
                std::uint64_t previous) noexcept
    -> std::expected<std::uint64_t, memory::fault> {
  const let loaded = space.load<T>({address & ~std::uint64_t{sizeof(T) - 1}});
  if (!loaded) {
    let fault = loaded.error();
    fault.address.value = address;
    return std::unexpected(fault);
  }
  const let offset = static_cast<unsigned>(address & (sizeof(T) - 1));
  const let shift = (Right ? sizeof(T) - 1 - offset : offset) * 8;
  const T mask = static_cast<T>(~T{0});
  const T keep = static_cast<T>(Right ? ~(mask >> shift) : ~(mask << shift));
  const T added = static_cast<T>(Right ? *loaded >> shift : *loaded << shift);
  const T value = (static_cast<T>(previous) & keep) | added;
  if constexpr (sizeof(T) == 4)
    return signed_word(value);
  else
    return value;
}

template <typename T, bool Right>
fn store_merge(memory::address_space &space, std::uint64_t address,
                 std::uint64_t value) noexcept
    -> std::expected<void, memory::fault> {
  const let offset = static_cast<unsigned>(address & (sizeof(T) - 1));
  const let begin = Right ? 0u : offset;
  const let count =
      Right ? offset + 1 : static_cast<unsigned>(sizeof(T)) - offset;
  std::array<std::byte, sizeof(T)> bytes;
  for (unsigned i = 0; i < count; i++) {
    const let shift =
        (Right ? count - 1 - i : static_cast<unsigned>(sizeof(T)) - 1 - i) * 8;
    bytes[i] = static_cast<std::byte>((value >> shift) & 255);
  }
  const let stored =
      space.store_bytes({(address & ~std::uint64_t{sizeof(T) - 1}) + begin},
                        std::span{bytes}.first(count));
  if (!stored) {
    let fault = stored.error();
    fault.address.value = address;
    fault.size = static_cast<memory::access_size>(sizeof(T));
    return std::unexpected(fault);
  }
  return {};
}
} // namespace detail

[[nodiscard]] fn
supports(std::span<const mips64::instruction_word> words) noexcept -> bool {
  if (words.empty() || words.size() > 65)
    return false;
  for (std::size_t i = 0; i < words.size(); i++) {
    if (!detail::supported(words[i]))
      return false;
    if (detail::control(words[i])) {
      if (i + 2 != words.size() || !detail::supported(words[i + 1]) ||
          detail::control(words[i + 1]))
        return false;
      return true;
    }
  }
  return true;
}

// One basic block. Source backing remains borrowed for the execution; a
// monitored write or external callback exits before the next ordinary slot.
[[nodiscard]] fn
execute(std::span<const mips64::instruction_word> words,
        mips64::guest_address start, mips64::state &state,
        memory::address_space &space,
        std::span<const std::span<const std::byte>> source_ranges = {}) noexcept
    -> std::expected<result, failure> {
  if (!supports(words) || start.value % 4 ||
      words.size() > (UINT64_MAX - start.value) / 4)
    return std::unexpected(
        failure{{}, start.value, {}, 0, error_code::unsupported});
  detail::execution progress{state.cop0};
  memory::write_monitor writes{source_ranges, false, &progress,
                               &detail::execution::before_external};
  const let monitoring = space.monitor_writes(&writes);
  std::optional<std::uint64_t> branch_pc;
  std::uint64_t next_pc = start.value + words.size() * 4;
  const let read = [&state](std::uint32_t index) {
    return index ? state.registers[index] : std::uint64_t{0};
  };
  const let write = [&state](std::uint32_t index, std::uint64_t value) {
    if (index)
      state.registers[index] = value;
  };
  for (std::size_t i = 0; i < words.size(); i++) {
    const let bits = words[i].bits;
    const let op = bits >> 26;
    const let rs = (bits >> 21) & 31;
    const let rt = (bits >> 16) & 31;
    const let rd = (bits >> 11) & 31;
    const let amount = (bits >> 6) & 31;
    const let pc = start.value + i * 4;
    const let left = read(rs);
    const let right = read(rt);
    const let immediate = static_cast<std::uint64_t>(std::int64_t{
        std::bit_cast<std::int16_t>(static_cast<std::uint16_t>(bits))});
    const let address = left + immediate;
    const let word_left = static_cast<std::uint32_t>(left);
    const let word_right = static_cast<std::uint32_t>(right);
    std::optional<memory::fault> fault;
    const let loaded = [&](std::expected<std::uint64_t, memory::fault> value) {
      if (value)
        write(rt, *value);
      else
        fault = value.error();
    };
    const let stored = [&](std::expected<void, memory::fault> value) {
      if (!value)
        fault = value.error();
    };
    if (detail::control(words[i])) {
      branch_pc = pc;
      bool taken = true;
      bool likely = false;
      std::uint64_t target = pc + 4 + (immediate << 2);
      if (op == 0) {
        target = left;
        if ((bits & 63) == 9)
          write(rd, pc + 8);
      } else if (op == 2 || op == 3) {
        target = ((pc + 4) & ~std::uint64_t{0x0fff'ffff}) |
                 (std::uint64_t{bits & 0x03ff'ffff} << 2);
        if (op == 3)
          write(31, pc + 8);
      } else if (op == 1) {
        taken = (rt & 1) ? std::bit_cast<std::int64_t>(left) >= 0
                         : std::bit_cast<std::int64_t>(left) < 0;
        likely = (rt & 2) != 0;
      } else {
        switch (op) {
        case 4:
        case 0x14:
          taken = left == right;
          break;
        case 5:
        case 0x15:
          taken = left != right;
          break;
        case 6:
          taken = std::bit_cast<std::int64_t>(left) <= 0;
          break;
        case 7:
          taken = std::bit_cast<std::int64_t>(left) > 0;
          break;
        default:
          break;
        }
        likely = op == 0x14 || op == 0x15;
      }
      next_pc = taken ? target : pc + 8;
      progress.completed++;
      if (likely && !taken)
        break;
      continue;
    }
    switch (op) {
    case 0:
      switch (bits & 63) {
      case 0:
        write(rd, detail::signed_word(word_right << amount));
        break;
      case 2:
        write(rd, detail::signed_word(word_right >> amount));
        break;
      case 3:
        write(rd, detail::signed_word(static_cast<std::uint32_t>(
                      std::bit_cast<std::int32_t>(word_right) >> amount)));
        break;
      case 4:
        write(rd, detail::signed_word(word_right << (left & 31)));
        break;
      case 6:
        write(rd, detail::signed_word(word_right >> (left & 31)));
        break;
      case 7:
        write(rd, detail::signed_word(static_cast<std::uint32_t>(
                      std::bit_cast<std::int32_t>(word_right) >> (left & 31))));
        break;
      case 0x14:
        write(rd, right << (left & 63));
        break;
      case 0x16:
        write(rd, right >> (left & 63));
        break;
      case 0x17:
        write(rd, static_cast<std::uint64_t>(
                      std::bit_cast<std::int64_t>(right) >> (left & 63)));
        break;
      case 0x21:
        write(rd, detail::signed_word(word_left + word_right));
        break;
      case 0x23:
        write(rd, detail::signed_word(word_left - word_right));
        break;
      case 0x24:
        write(rd, left & right);
        break;
      case 0x25:
        write(rd, left | right);
        break;
      case 0x26:
        write(rd, left ^ right);
        break;
      case 0x27:
        write(rd, ~(left | right));
        break;
      case 0x2a:
        write(rd, std::bit_cast<std::int64_t>(left) <
                      std::bit_cast<std::int64_t>(right));
        break;
      case 0x2b:
        write(rd, left < right);
        break;
      case 0x2d:
        write(rd, left + right);
        break;
      case 0x2f:
        write(rd, left - right);
        break;
      case 0x38:
        write(rd, right << amount);
        break;
      case 0x3a:
        write(rd, right >> amount);
        break;
      case 0x3b:
        write(rd, static_cast<std::uint64_t>(
                      std::bit_cast<std::int64_t>(right) >> amount));
        break;
      case 0x3c:
        write(rd, right << (amount + 32));
        break;
      case 0x3e:
        write(rd, right >> (amount + 32));
        break;
      case 0x3f:
        write(rd, static_cast<std::uint64_t>(
                      std::bit_cast<std::int64_t>(right) >> (amount + 32)));
        break;
      default:
        break;
      }
      break;
    case 9:
      write(rt, detail::signed_word(word_left +
                                    static_cast<std::uint32_t>(immediate)));
      break;
    case 0x0a:
      write(rt, std::bit_cast<std::int64_t>(left) <
                    std::bit_cast<std::int64_t>(immediate));
      break;
    case 0x0b:
      write(rt, left < immediate);
      break;
    case 0x0c:
      write(rt, left & (bits & 0xffff));
      break;
    case 0x0d:
      write(rt, left | (bits & 0xffff));
      break;
    case 0x0e:
      write(rt, left ^ (bits & 0xffff));
      break;
    case 0x0f:
      write(rt, detail::signed_word((bits & 0xffff) << 16));
      break;
    case 0x19:
      write(rt, address);
      break;
    case 0x1a:
      loaded(detail::load_merge<std::uint64_t, false>(space, address, right));
      break;
    case 0x1b:
      loaded(detail::load_merge<std::uint64_t, true>(space, address, right));
      break;
    case 0x20:
      loaded(detail::load<std::uint8_t, true>(space, address));
      break;
    case 0x21:
      loaded(detail::load<std::uint16_t, true>(space, address));
      break;
    case 0x22:
      loaded(detail::load_merge<std::uint32_t, false>(space, address, right));
      break;
    case 0x23:
      loaded(detail::load<std::uint32_t, true>(space, address));
      break;
    case 0x24:
      loaded(detail::load<std::uint8_t>(space, address));
      break;
    case 0x25:
      loaded(detail::load<std::uint16_t>(space, address));
      break;
    case 0x26:
      loaded(detail::load_merge<std::uint32_t, true>(space, address, right));
      break;
    case 0x27:
      loaded(detail::load<std::uint32_t>(space, address));
      break;
    case 0x28:
      stored(space.store<std::uint8_t>({address},
                                       static_cast<std::uint8_t>(right)));
      break;
    case 0x29:
      stored(space.store<std::uint16_t>({address},
                                        static_cast<std::uint16_t>(right)));
      break;
    case 0x2a:
      stored(detail::store_merge<std::uint32_t, false>(space, address, right));
      break;
    case 0x2b:
      stored(space.store<std::uint32_t>({address}, word_right));
      break;
    case 0x2c:
      stored(detail::store_merge<std::uint64_t, false>(space, address, right));
      break;
    case 0x2d:
      stored(detail::store_merge<std::uint64_t, true>(space, address, right));
      break;
    case 0x2e:
      stored(detail::store_merge<std::uint32_t, true>(space, address, right));
      break;
    case 0x30: {
      const let value = space.load_linked<std::uint32_t>({address});
      if (value)
        write(rt, detail::signed_word(*value));
      else
        fault = value.error();
      break;
    }
    case 0x37:
      loaded(detail::load<std::uint64_t>(space, address));
      break;
    case 0x38: {
      const let value =
          space.store_conditional<std::uint32_t>({address}, word_right);
      if (value)
        write(rt, *value ? 1u : 0u);
      else
        fault = value.error();
      break;
    }
    case 0x3f:
      stored(space.store<std::uint64_t>({address}, right));
      break;
    default:
      break;
    }
    if (fault)
      return std::unexpected(failure{*fault, pc, branch_pc,
                                     progress.completed - progress.accounted,
                                     error_code::memory, progress.completed});
    progress.completed++;
    if (writes.modified && !branch_pc)
      return result{pc + 4, progress.completed, progress.accounted};
  }
  return result{next_pc, progress.completed, progress.accounted};
}
} // namespace jit::interpreter
