module;

#include <bit>
#include <concepts>
#include <cstddef>
#include <variant>
#include <cstdint>
#include <expected>
#include <format>
#include <functional>
#include <limits>
#include <memory>
#include <memory_resource>
#include <new>
#include <optional>
#include <print>
#include <span>
#include <string_view>
#include <utility>
#include <type_traits>
#include <vector>

#include <dynamips/macros.h>

export module jit.ir;

export namespace jit::ir {

struct value_id {
  std::uint32_t index = std::numeric_limits<std::uint32_t>::max();
};

struct state_slot {
  std::uint16_t index;
};

enum class width : std::uint8_t { bits32 = 32, bits64 = 64 };
enum class opcode : std::uint8_t {
  syscall,
  constant,
  read_hilo,
  write_hilo,
  read_state,
  shift_right_arithmetic32,
  shift_right32,
  shift_left32,
  shift_left64,
  shift_right64,
  shift_right_arithmetic64,
  sign_extend32,
  unsigned_less,
  signed_less,
  add_wrap,
  bit_and,
  bit_or,
  bit_xor,
  write_state,
  write_cop0,
  read_cop0,
  read_cop0_64,
  write_cop0_64,
  load8_signed,
  load32_signed,
  load64,
  store8,
  store32,
  store64,
};

constexpr fn produces_value(opcode op) noexcept -> bool {
  return op != opcode::write_hilo && op != opcode::syscall &&
         op != opcode::write_cop0_64 && op != opcode::write_cop0 &&
         op != opcode::write_state && op != opcode::store8 &&
         op != opcode::store32 && op != opcode::store64;
}

enum class binary_kind : std::uint8_t {
  unsigned_less,
  signed_less,
  add_wrap,
  bit_and,
  bit_or,
  bit_xor
};

struct operands {
  value_id left;
  value_id right;
};

struct shift_operand {
  value_id value;
  std::uint32_t amount;
};

union payload {
  shift_operand shift;
  std::uint64_t immediate;
  operands inputs;
};

struct node {
  payload data;
  std::uint32_t source_index;
  state_slot slot;
  opcode op;
  width type;

  static constexpr fn syscall(std::uint32_t source) noexcept -> node {
    return {{.immediate = 0}, source, {}, opcode::syscall, width::bits64};
  }

  static constexpr fn constant(std::uint64_t bits, std::uint32_t source,
                                width type = width::bits64) noexcept -> node {
    return {{.immediate = bits}, source, {}, opcode::constant, type};
  }

  static constexpr fn read_hilo(state_slot slot, std::uint32_t source) noexcept
      -> node {
    return {{.immediate = 0}, source, slot, opcode::read_hilo, width::bits64};
  }

  static constexpr fn write_hilo(state_slot slot, value_id value,
                                  std::uint32_t source) noexcept -> node {
    return {{.inputs = {value, {}}},
            source,
            slot,
            opcode::write_hilo,
            width::bits64};
  }

  static constexpr fn read(state_slot slot, std::uint32_t source) noexcept
      -> node {
    return {{.immediate = 0}, source, slot, opcode::read_state, width::bits64};
  }

  static constexpr fn shift_right_arithmetic32(value_id value,
                                                std::uint32_t amount,
                                                std::uint32_t source) noexcept
      -> node {
    return {{.shift = {value, amount}},
            source,
            {},
            opcode::shift_right_arithmetic32,
            width::bits64};
  }

  static constexpr fn shift_right32(value_id value, std::uint32_t amount,
                                     std::uint32_t source) noexcept -> node {
    return {{.shift = {value, amount}},
            source,
            {},
            opcode::shift_right32,
            width::bits64};
  }

  static constexpr fn shift_left32(value_id value, std::uint32_t amount,
                                    std::uint32_t source) noexcept -> node {
    return {{.shift = {value, amount}},
            source,
            {},
            opcode::shift_left32,
            width::bits64};
  }

  static constexpr fn shift_left64(value_id value, std::uint32_t amount,
                                    std::uint32_t source) noexcept -> node {
    return {{.shift = {value, amount}},
            source,
            {},
            opcode::shift_left64,
            width::bits64};
  }

  static constexpr fn shift_right64(value_id value, std::uint32_t amount,
                                     std::uint32_t source) noexcept -> node {
    return {{.shift = {value, amount}},
            source,
            {},
            opcode::shift_right64,
            width::bits64};
  }

  static constexpr fn shift_right_arithmetic64(value_id value,
                                                std::uint32_t amount,
                                                std::uint32_t source) noexcept
      -> node {
    return {{.shift = {value, amount}},
            source,
            {},
            opcode::shift_right_arithmetic64,
            width::bits64};
  }

  static constexpr fn sign_extend32(value_id value,
                                     std::uint32_t source) noexcept -> node {
    return {{.inputs = {value, {}}},
            source,
            {},
            opcode::sign_extend32,
            width::bits64};
  }

  static constexpr fn binary(binary_kind kind, value_id left, value_id right,
                              std::uint32_t source,
                              width type = width::bits64) noexcept -> node {
    const let op = kind == binary_kind::unsigned_less ? opcode::unsigned_less
                   : kind == binary_kind::signed_less ? opcode::signed_less
                   : kind == binary_kind::add_wrap    ? opcode::add_wrap
                   : kind == binary_kind::bit_and     ? opcode::bit_and
                   : kind == binary_kind::bit_or      ? opcode::bit_or
                                                      : opcode::bit_xor;
    return {{.inputs = {left, right}}, source, {}, op, type};
  }

  static constexpr fn write(state_slot slot, value_id value,
                             std::uint32_t source) noexcept -> node {
    return {{.inputs = {value, {}}},
            source,
            slot,
            opcode::write_state,
            width::bits64};
  }

  static constexpr fn read_cop0(state_slot slot, std::uint32_t source) noexcept
      -> node {
    return {{.immediate = 0}, source, slot, opcode::read_cop0, width::bits64};
  }

  static constexpr fn write_cop0(state_slot slot, value_id value,
                                  std::uint32_t source) noexcept -> node {
    return {{.inputs = {value, {}}},
            source,
            slot,
            opcode::write_cop0,
            width::bits64};
  }

  static constexpr fn read_cop0_64(state_slot slot,
                                    std::uint32_t source) noexcept -> node {
    return {
        {.immediate = 0}, source, slot, opcode::read_cop0_64, width::bits64};
  }

  static constexpr fn write_cop0_64(state_slot slot, value_id value,
                                     std::uint32_t source) noexcept -> node {
    return {{.inputs = {value, {}}},
            source,
            slot,
            opcode::write_cop0_64,
            width::bits64};
  }

  static constexpr fn load8_signed(value_id address,
                                    std::uint32_t source) noexcept -> node {
    return {{.inputs = {address, {}}},
            source,
            {},
            opcode::load8_signed,
            width::bits64};
  }

  static constexpr fn load(value_id address, std::uint32_t source,
                            width size) noexcept -> node {
    return {{.inputs = {address, {}}},
            source,
            {},
            size == width::bits32 ? opcode::load32_signed : opcode::load64,
            width::bits64};
  }

  static constexpr fn store8(value_id address, value_id value,
                              std::uint32_t source) noexcept -> node {
    return {{.inputs = {address, value}},
            source,
            {},
            opcode::store8,
            width::bits64};
  }

  static constexpr fn store(value_id address, value_id value,
                             std::uint32_t source, width size) noexcept
      -> node {
    return {{.inputs = {address, value}},
            source,
            {},
            size == width::bits32 ? opcode::store32 : opcode::store64,
            width::bits64};
  }
};

static_assert(sizeof(node) == 16);
static_assert(std::is_trivially_copyable_v<node>);

enum class error_code {
  invalid_metadata,
  invalid_type,
  invalid_operand,
  invalid_slot,
  invalid_opcode,
  block_too_large,
  capacity_exhausted,
  out_of_memory,
};

struct error {
  error_code code;
  std::size_t node_index;
};

struct block_info {
  std::uint64_t start_pc;
  std::uint32_t instruction_count;
  std::uint16_t state_slots;
};

enum class branch_kind : std::uint8_t { equal, not_equal };

struct branch_exit {
  value_id left;
  value_id right;
  std::uint64_t target;
  std::uint32_t source_index;
  branch_kind kind;
  bool annul_not_taken = false;
};

struct jump_exit {
  std::uint64_t target;
  std::uint32_t source_index;
};

struct indirect_exit {
  value_id target;
  std::uint32_t source_index;
};

using block_exit =
    std::variant<std::monostate, branch_exit, jump_exit, indirect_exit>;

class block {
public:
  block(const block &) = delete;
  fn operator=(const block &) -> block & = delete;
  block(block &&) noexcept = default;
  fn operator=(block &&other) noexcept -> block & {
    if (this != &other) {
      // reconstruction transfers PMR storage without allocating
      info_ = other.info_;
      exit_ = other.exit_;
      std::destroy_at(&nodes_);
      std::construct_at(&nodes_, std::move(other.nodes_));
    }
    return *this;
  }

  [[nodiscard]] static fn create(block_info info, std::size_t capacity,
                                  std::pmr::memory_resource &memory) noexcept
      -> std::expected<block, error> {
    block result{info, memory};
    if (let valid = result.verify(); !valid)
      return std::unexpected(valid.error());
    if (capacity > std::numeric_limits<std::uint32_t>::max() ||
        capacity > result.nodes_.max_size())
      return std::unexpected(error{error_code::block_too_large, 0});
    try {
      result.nodes_.reserve(capacity);
    } catch (const std::bad_alloc &) {
      return std::unexpected(error{error_code::out_of_memory, 0});
    }
    return result;
  }

  [[nodiscard]] fn append(node value) noexcept
      -> std::expected<value_id, error> {
    const let index = nodes_.size();
    if (index >= std::numeric_limits<std::uint32_t>::max() ||
        index == nodes_.max_size())
      return std::unexpected(error{error_code::block_too_large, index});
    if (index == nodes_.capacity())
      return std::unexpected(error{error_code::capacity_exhausted, index});
    if (let valid = verify_node(value, index); !valid)
      return std::unexpected(valid.error());
    nodes_.push_back(value);
    return value_id{static_cast<std::uint32_t>(index)};
  }

  [[nodiscard]] fn nodes() const noexcept -> std::span<const node> {
    return nodes_;
  }

  [[nodiscard]] fn edit_nodes() noexcept -> std::span<node> { return nodes_; }

  [[nodiscard]] fn info() const noexcept -> block_info { return info_; }

  fn branch() const noexcept -> const branch_exit * {
    return std::get_if<branch_exit>(&exit_);
  }

  fn jump() const noexcept -> const jump_exit * {
    return std::get_if<jump_exit>(&exit_);
  }

  fn indirect() const noexcept -> const indirect_exit * {
    return std::get_if<indirect_exit>(&exit_);
  }

  fn branch_pc() const noexcept -> std::optional<std::uint64_t> {
    if (const let *value = branch())
      return info_.start_pc + std::uint64_t{value->source_index} * 4;
    if (const let *value = jump())
      return info_.start_pc + std::uint64_t{value->source_index} * 4;
    if (const let *value = indirect())
      return info_.start_pc + std::uint64_t{value->source_index} * 4;
    return std::nullopt;
  }

  [[nodiscard]] fn set_exit(block_exit exit) noexcept
      -> std::expected<void, error> {
    const let previous = exit_;
    exit_ = exit;
    if (let result = verify(); !result) {
      exit_ = previous;
      return result;
    }
    return {};
  }

  [[nodiscard]] fn set_branch(branch_exit exit) noexcept
      -> std::expected<void, error> {
    return set_exit(exit);
  }

  [[nodiscard]] fn set_jump(jump_exit exit) noexcept
      -> std::expected<void, error> {
    return set_exit(exit);
  }

  [[nodiscard]] fn set_indirect(indirect_exit exit) noexcept
      -> std::expected<void, error> {
    return set_exit(exit);
  }

  [[nodiscard]] fn source_pc(const node &value) const noexcept
      -> std::uint64_t {
    return info_.start_pc + std::uint64_t{value.source_index} * 4;
  }

  [[nodiscard]] fn next_pc() const noexcept -> std::uint64_t {
    return info_.start_pc + std::uint64_t{info_.instruction_count} * 4;
  }

  [[nodiscard]] fn verify() const noexcept -> std::expected<void, error> {
    if (info_.start_pc % 4 != 0 ||
        info_.instruction_count >
            (std::numeric_limits<std::uint64_t>::max() - info_.start_pc) / 4)
      return std::unexpected(error{error_code::invalid_metadata, 0});
    if (const let *exit = indirect()) {
      if (info_.instruction_count < 2 ||
          exit->source_index != info_.instruction_count - 2)
        return std::unexpected(error{error_code::invalid_metadata, 0});
      if (exit->target.index >= nodes_.size() ||
          !produces_value(nodes_[exit->target.index].op) ||
          nodes_[exit->target.index].type != width::bits64 ||
          nodes_[exit->target.index].source_index > exit->source_index)
        return std::unexpected(
            error{error_code::invalid_operand, exit->target.index});
    }
    if (const let *exit = jump()) {
      if (info_.instruction_count < 2 ||
          exit->source_index != info_.instruction_count - 2 ||
          exit->target % 4 != 0)
        return std::unexpected(error{error_code::invalid_metadata, 0});
    }
    if (branch()) {
      const let &exit = *branch();
      if (info_.instruction_count < 2 ||
          exit.source_index != info_.instruction_count - 2 ||
          exit.target % 4 != 0 ||
          (exit.kind != branch_kind::equal &&
           exit.kind != branch_kind::not_equal))
        return std::unexpected(error{error_code::invalid_metadata, 0});
      if (exit.annul_not_taken) {
        bool in_slot = false;
        for (const let &value : nodes_) {
          if (value.source_index > exit.source_index)
            in_slot = true;
          else if (in_slot)
            return std::unexpected(error{error_code::invalid_metadata, 0});
        }
      }
      for (const let input : {exit.left, exit.right})
        if (input.index >= nodes_.size() ||
            !produces_value(nodes_[input.index].op) ||
            nodes_[input.index].type != width::bits64 ||
            nodes_[input.index].source_index > exit.source_index)
          return std::unexpected(
              error{error_code::invalid_operand, input.index});
    }
    for (std::size_t i = 0; i < nodes_.size(); i++)
      if (let valid = verify_node(nodes_[i], i); !valid)
        return valid;
    return {};
  }

  fn print(std::string_view label = {}) const -> void;

private:
  friend struct std::formatter<block>;

  explicit block(block_info info, std::pmr::memory_resource &memory) noexcept
      : info_(info), nodes_(&memory) {}

  block_info info_;
  block_exit exit_;
  std::pmr::vector<node> nodes_;

  fn verify_node(const node &current, std::size_t index) const noexcept
      -> std::expected<void, error> {
    const let fail = [&](error_code code) {
      return std::unexpected(error{code, index});
    };
    if (current.source_index >= info_.instruction_count)
      return fail(error_code::invalid_metadata);
    if (current.type != width::bits32 && current.type != width::bits64)
      return fail(error_code::invalid_type);
    const let operand_valid = [&](value_id input) {
      return input.index < index && produces_value(nodes_[input.index].op) &&
             nodes_[input.index].type == current.type;
    };
    switch (current.op) {
    case opcode::syscall:
      if (current.type != width::bits64)
        return fail(error_code::invalid_type);
      break;
    case opcode::constant:
      if (current.type == width::bits32 &&
          current.data.immediate > std::numeric_limits<std::uint32_t>::max())
        return fail(error_code::invalid_operand);
      break;
    case opcode::read_hilo:
    case opcode::write_hilo:
      if (current.slot.index > 1)
        return fail(error_code::invalid_slot);
      if (current.type != width::bits64)
        return fail(error_code::invalid_type);
      if (current.op == opcode::write_hilo &&
          !operand_valid(current.data.inputs.left))
        return fail(error_code::invalid_operand);
      break;
    case opcode::read_cop0_64:
    case opcode::write_cop0_64:
    case opcode::read_cop0:
    case opcode::write_cop0:
      if (current.slot.index != 9 && current.slot.index != 11 &&
          current.slot.index != 12 && current.slot.index != 13 &&
          current.slot.index != 15 && current.slot.index != 16 &&
          current.slot.index != 8 && current.slot.index != 14 &&
          current.slot.index != 30)
        return fail(error_code::invalid_slot);
      if (current.type != width::bits64)
        return fail(error_code::invalid_type);
      if ((current.op == opcode::write_cop0 ||
           current.op == opcode::write_cop0_64) &&
          !operand_valid(current.data.inputs.left))
        return fail(error_code::invalid_operand);
      break;
    case opcode::read_state:
    case opcode::write_state:
      if (current.slot.index >= info_.state_slots)
        return fail(error_code::invalid_slot);
      if (current.type != width::bits64)
        return fail(error_code::invalid_type);
      if (current.op == opcode::write_state &&
          !operand_valid(current.data.inputs.left))
        return fail(error_code::invalid_operand);
      break;
    case opcode::shift_left64:
    case opcode::shift_right64:
    case opcode::shift_right_arithmetic64:
      if (current.type != width::bits64)
        return fail(error_code::invalid_type);
      if (!operand_valid(current.data.shift.value) ||
          current.data.shift.amount > 63)
        return fail(error_code::invalid_operand);
      break;
    case opcode::shift_right_arithmetic32:
    case opcode::shift_left32:
    case opcode::shift_right32:
      if (current.type != width::bits64)
        return fail(error_code::invalid_type);
      if (!operand_valid(current.data.shift.value) ||
          current.data.shift.amount > 31)
        return fail(error_code::invalid_operand);
      break;
    case opcode::sign_extend32:
      if (current.type != width::bits64)
        return fail(error_code::invalid_type);
      if (!operand_valid(current.data.inputs.left))
        return fail(error_code::invalid_operand);
      break;
    case opcode::unsigned_less:
    case opcode::signed_less:
      if (current.type != width::bits64)
        return fail(error_code::invalid_type);
      [[fallthrough]];
    case opcode::add_wrap:
    case opcode::bit_and:
    case opcode::bit_or:
    case opcode::bit_xor:
      if (!operand_valid(current.data.inputs.left) ||
          !operand_valid(current.data.inputs.right))
        return fail(error_code::invalid_operand);
      break;
    case opcode::load8_signed:
    case opcode::load32_signed:
    case opcode::load64:
    case opcode::store8:
    case opcode::store32:
    case opcode::store64:
      if (current.type != width::bits64)
        return fail(error_code::invalid_type);
      if (!operand_valid(current.data.inputs.left) ||
          (!produces_value(current.op) &&
           !operand_valid(current.data.inputs.right)))
        return fail(error_code::invalid_operand);
      break;
    default:
      return fail(error_code::invalid_opcode);
    }
    return {};
  }

  template <typename Output> fn format_to(Output out) const -> Output {
    if (!verify())
      throw std::format_error("cannot print invalid IR");
    out = std::format_to(out,
                         "; {} nodes, {} bytes/node, {} bytes reserved\n"
                         "define i64 @block_{:x}(ptr %state) {{\nentry:\n",
                         nodes_.size(), sizeof(node),
                         nodes_.capacity() * sizeof(node), info_.start_pc);
    for (std::size_t i = 0; i < nodes_.size(); i++) {
      const let &current = nodes_[i];
      const let bits = static_cast<unsigned>(current.type);
      switch (current.op) {
      case opcode::syscall:
        out = std::format_to(out, "  raise.syscall");
        break;
      case opcode::constant:
        out = std::format_to(out, "  %v{} = const i{} {}", i, bits,
                             current.data.immediate);
        break;
      case opcode::shift_left64:
      case opcode::shift_right64:
      case opcode::shift_right_arithmetic64:
        out = std::format_to(out, "  %v{} = {} %v{}, {}", i,
                             current.op == opcode::shift_left64    ? "shl.i64"
                             : current.op == opcode::shift_right64 ? "lshr.i64"
                                                                   : "ashr.i64",
                             current.data.shift.value.index,
                             current.data.shift.amount);
        break;
      case opcode::shift_right_arithmetic32:
      case opcode::shift_left32:
      case opcode::shift_right32:
        out = std::format_to(
            out, "  %v{} = {} %v{}, {}", i,
            current.op == opcode::shift_left32    ? "shl.i32.sext"
            : current.op == opcode::shift_right32 ? "lshr.i32.sext"
                                                  : "ashr.i32.sext",
            current.data.shift.value.index, current.data.shift.amount);
        break;
      case opcode::sign_extend32:
        out = std::format_to(out, "  %v{} = sext.i32 i64 %v{}", i,
                             current.data.inputs.left.index);
        break;
      case opcode::read_state:
        out = std::format_to(out, "  %v{} = state.read i64 %state, {}", i,
                             current.slot.index);
        break;
      case opcode::read_hilo:
        out = std::format_to(out, "  %v{} = {}.read i64", i,
                             current.slot.index == 0 ? "hi" : "lo");
        break;
      case opcode::write_hilo:
        out = std::format_to(out, "  {}.write i64 %v{}",
                             current.slot.index == 0 ? "hi" : "lo",
                             current.data.inputs.left.index);
        break;
      case opcode::read_cop0_64:
      case opcode::read_cop0:
        out = std::format_to(out, "  %v{} = cop0.read.{} {}", i,
                             current.op == opcode::read_cop0_64 ? "i64" : "s32",
                             current.slot.index);
        break;
      case opcode::write_cop0_64:
      case opcode::write_cop0:
        out =
            std::format_to(out, "  cop0.write.{} %v{}, {}",
                           current.op == opcode::write_cop0_64 ? "i64" : "i32",
                           current.data.inputs.left.index, current.slot.index);
        break;
      case opcode::write_state:
        out =
            std::format_to(out, "  state.write i64 %v{}, %state, {}",
                           current.data.inputs.left.index, current.slot.index);
        break;
      case opcode::load8_signed:
      case opcode::load32_signed:
      case opcode::load64:
        out = std::format_to(out, "  %v{} = load.guest.{} %v{}", i,
                             current.op == opcode::load8_signed    ? "s8"
                             : current.op == opcode::load32_signed ? "s32"
                                                                   : "i64",
                             current.data.inputs.left.index);
        break;
      case opcode::store8:
      case opcode::store32:
      case opcode::store64:
        out = std::format_to(out, "  store.guest i{} %v{}, %v{}",
                             current.op == opcode::store8    ? 8
                             : current.op == opcode::store32 ? 32
                                                             : 64,
                             current.data.inputs.right.index,
                             current.data.inputs.left.index);
        break;
      case opcode::unsigned_less:
      case opcode::signed_less:
      case opcode::add_wrap:
      case opcode::bit_and:
      case opcode::bit_or:
      case opcode::bit_xor: {
        const let name = current.op == opcode::unsigned_less ? "icmp.ult"
                         : current.op == opcode::signed_less ? "icmp.slt"
                         : current.op == opcode::add_wrap    ? "add"
                         : current.op == opcode::bit_and     ? "and"
                         : current.op == opcode::bit_or      ? "or"
                                                             : "xor";
        out = std::format_to(out, "  %v{} = {} i{} %v{}, %v{}", i, name, bits,
                             current.data.inputs.left.index,
                             current.data.inputs.right.index);
        break;
      }
      default:
        throw std::format_error("cannot print invalid IR opcode");
      }
      out = std::format_to(out, " ; pc={:#x}\n", source_pc(current));
    }
    if (const let *exit = indirect())
      return std::format_to(
          out, "  jump i64 %v{} ; branch PC={:#x}, delay slot included\n}}",
          exit->target.index, *branch_pc());
    if (const let *exit = jump())
      return std::format_to(
          out, "  jump {:#x} ; branch PC={:#x}, delay slot included\n}}",
          exit->target, *branch_pc());
    if (branch())
      return std::format_to(
          out,
          "  br.{} %v{}, %v{}, {:#x}, {:#x} ; branch PC={:#x}, delay slot "
          "{}\n}}",
          branch()->kind == branch_kind::equal ? "eq" : "ne",
          branch()->left.index, branch()->right.index, branch()->target,
          next_pc(), info_.start_pc + std::uint64_t{branch()->source_index} * 4,
          branch()->annul_not_taken ? "annulled when not taken" : "included");
    return std::format_to(out, "  ret i64 {}\n}}", next_pc());
  }
};

[[nodiscard]] fn fold_constants(block &value) noexcept
    -> std::expected<std::size_t, error> {
  if (let valid = value.verify(); !valid)
    return std::unexpected(valid.error());
  std::size_t folded = 0;
  for (let &current : value.edit_nodes()) {
    if (current.op == opcode::shift_left64 ||
        current.op == opcode::shift_right64 ||
        current.op == opcode::shift_right_arithmetic64) {
      const let &input = value.nodes()[current.data.shift.value.index];
      if (input.op == opcode::constant) {
        const let bits = input.data.immediate;
        const let amount = current.data.shift.amount;
        const let shifted =
            current.op == opcode::shift_left64 ? bits << amount
            : current.op == opcode::shift_right64
                ? bits >> amount
                : std::bit_cast<std::uint64_t>(
                      std::bit_cast<std::int64_t>(bits) >> amount);
        current = node::constant(shifted, current.source_index);
        folded++;
      }
      continue;
    }
    if (current.op == opcode::shift_left32 ||
        current.op == opcode::shift_right32 ||
        current.op == opcode::shift_right_arithmetic32) {
      const let &input = value.nodes()[current.data.shift.value.index];
      if (input.op == opcode::constant) {
        const let input32 = static_cast<std::uint32_t>(input.data.immediate);
        const let word = current.op == opcode::shift_left32
                             ? input32 << current.data.shift.amount
                         : current.op == opcode::shift_right32
                             ? input32 >> current.data.shift.amount
                             : std::bit_cast<std::uint32_t>(
                                   std::bit_cast<std::int32_t>(input32) >>
                                   current.data.shift.amount);
        current = node::constant(static_cast<std::uint64_t>(std::int64_t{
                                     std::bit_cast<std::int32_t>(word)}),
                                 current.source_index);
        folded++;
      }
      continue;
    }
    if (current.op == opcode::sign_extend32) {
      const let &input = value.nodes()[current.data.inputs.left.index];
      if (input.op == opcode::constant) {
        const let bits =
            static_cast<std::uint64_t>(std::int64_t{static_cast<std::int32_t>(
                static_cast<std::uint32_t>(input.data.immediate))});
        current = node::constant(bits, current.source_index);
        folded++;
      }
      continue;
    }
    if (current.op != opcode::add_wrap && current.op != opcode::bit_or &&
        current.op != opcode::bit_xor && current.op != opcode::bit_and &&
        current.op != opcode::signed_less &&
        current.op != opcode::unsigned_less)
      continue;
    const let &left = value.nodes()[current.data.inputs.left.index];
    const let &right = value.nodes()[current.data.inputs.right.index];
    if (left.op != opcode::constant || right.op != opcode::constant)
      continue;
    std::uint64_t bits = 0;
    switch (current.op) {
    case opcode::unsigned_less:
      bits = left.data.immediate < right.data.immediate;
      break;
    case opcode::signed_less:
      bits = std::bit_cast<std::int64_t>(left.data.immediate) <
             std::bit_cast<std::int64_t>(right.data.immediate);
      break;
    case opcode::add_wrap:
      bits = left.data.immediate + right.data.immediate;
      break;
    case opcode::bit_and:
      bits = left.data.immediate & right.data.immediate;
      break;
    case opcode::bit_or:
      bits = left.data.immediate | right.data.immediate;
      break;
    case opcode::bit_xor:
      bits = left.data.immediate ^ right.data.immediate;
      break;
    default:
      break;
    }
    if (current.type == width::bits32)
      bits &= std::numeric_limits<std::uint32_t>::max();
    current = node::constant(bits, current.source_index, current.type);
    folded++;
  }
  return folded;
}

template <typename Pass>
concept optimization_pass = requires(Pass &pass, block &value) {
  {
    std::invoke(pass, value)
  } noexcept -> std::same_as<std::expected<std::size_t, error>>;
};

template <optimization_pass First, optimization_pass... Rest>
[[nodiscard]] fn optimize(block &value, First &&first, Rest &&...rest) noexcept
    -> std::expected<std::size_t, error> {
  if (let valid = value.verify(); !valid)
    return std::unexpected(valid.error());
  std::expected<std::size_t, error> total{0};
  const let apply = [&](auto &&pass) {
    if (!total)
      return;
    const let changed = std::invoke(pass, value);
    if (!changed) {
      total = std::unexpected(changed.error());
      return;
    }
    if (let valid = value.verify(); !valid) {
      total = std::unexpected(valid.error());
      return;
    }
    *total += *changed;
  };
  apply(first);
  (apply(rest), ...);
  return total;
}

} // namespace jit::ir

export template <> struct std::formatter<jit::ir::block> {
  constexpr fn parse(std::format_parse_context &context) {
    const let it = context.begin();
    if (it != context.end() && *it != '}')
      throw std::format_error("IR blocks do not take format specifiers");
    return it;
  }

  template <typename Context>
  fn format(const jit::ir::block &value, Context &context) const {
    return value.format_to(context.out());
  }
};

namespace jit::ir {
fn block::print(std::string_view label) const -> void {
  if (!label.empty())
    std::println("; {}", label);
  std::println("{}", *this);
}
} // namespace jit::ir
