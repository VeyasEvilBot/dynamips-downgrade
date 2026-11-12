module;

#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <memory>
#include <memory_resource>
#include <new>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <dynamips/macros.h>

export module jit.amd64;

import asmjit;
import jit.mips64;
import jit.ir;
import jit.memory;

export namespace jit::amd64 {

enum class error_code {
  invalid_ir,
  out_of_memory,
  code_generation,
  invalid_state,
  unavailable,
  memory_required,
  cop0_required,
  hilo_required,
  memory_fault,
};

struct error {
  error_code code;
  std::size_t node_index = 0;
  std::uint32_t native_code = 0;
};

struct execution_error {
  error_code code;
  std::uint64_t pc;
  memory::fault access{};
  std::optional<std::uint64_t> branch_pc{};
};

[[nodiscard]] fn message(error value) noexcept -> std::string_view {
  switch (value.code) {
  case error_code::invalid_ir:
    return "invalid IR";
  case error_code::out_of_memory:
    return "compilation scratch or native allocation exhausted";
  case error_code::code_generation:
    return value.native_code
               ? asmjit::DebugUtils::errorAsString(value.native_code)
               : "code generation failed";
  case error_code::invalid_state:
    return "state has too few slots";
  case error_code::unavailable:
    return "native AMD64 backend is unavailable";
  case error_code::hilo_required:
    return "block requires HI/LO state";
  case error_code::cop0_required:
    return "block requires CP0 state";
  case error_code::memory_required:
    return "block requires guest memory";
  case error_code::memory_fault:
    return "guest memory access failed";
  }
  return "native compilation failed";
}

class compiler;

class compiled_block {
public:
  compiled_block(const compiled_block &) = delete;
  fn operator=(const compiled_block &) -> compiled_block & = delete;

  compiled_block(compiled_block &&other) noexcept
      : runtime_(std::move(other.runtime_)),
        entry_(std::exchange(other.entry_, nullptr)), slots_(other.slots_),
        uses_memory_(other.uses_memory_), uses_cop0_(other.uses_cop0_),
        uses_hilo_(other.uses_hilo_), branch_pc_(other.branch_pc_) {}

  fn operator=(compiled_block &&other) noexcept -> compiled_block & {
    if (this != &other) {
      reset();
      runtime_ = std::move(other.runtime_);
      entry_ = std::exchange(other.entry_, nullptr);
      slots_ = other.slots_;
      uses_memory_ = other.uses_memory_;
      uses_cop0_ = other.uses_cop0_;
      uses_hilo_ = other.uses_hilo_;
      branch_pc_ = other.branch_pc_;
    }
    return *this;
  }

  ~compiled_block() { reset(); }

  [[nodiscard]] fn run(std::span<std::uint64_t> state) const noexcept
      -> std::expected<std::uint64_t, error> {
    if (!entry_)
      return std::unexpected(error{error_code::unavailable});
    if (state.size() < slots_)
      return std::unexpected(error{error_code::invalid_state});
    if (uses_hilo_)
      return std::unexpected(error{error_code::hilo_required});
    if (uses_cop0_)
      return std::unexpected(error{error_code::cop0_required});
    if (uses_memory_)
      return std::unexpected(error{error_code::memory_required});
    return entry_(state.data(), nullptr);
  }

  [[nodiscard]] fn run(std::span<std::uint64_t> state,
                        memory::address_space &memory,
                        mips64::cop0_state *cop0 = nullptr,
                        mips64::hilo_state *hilo = nullptr) const noexcept
      -> std::expected<std::uint64_t, execution_error> {
    if (!entry_)
      return std::unexpected(execution_error{error_code::unavailable, 0});
    if (state.size() < slots_)
      return std::unexpected(execution_error{error_code::invalid_state, 0});
    if (uses_cop0_ && !cop0)
      return std::unexpected(execution_error{error_code::cop0_required, 0});
    if (uses_hilo_ && !hilo)
      return std::unexpected(execution_error{error_code::hilo_required, 0});
    execution_context context{&memory, cop0, hilo};
    const let pc = entry_(state.data(), &context);
    if (context.failed)
      return std::unexpected(execution_error{
          error_code::memory_fault, pc, context.fault,
          branch_pc_ && pc == *branch_pc_ + 4 ? branch_pc_ : std::nullopt});
    return pc;
  }

private:
  friend class compiler;

  struct execution_context {
    memory::address_space *memory;
    mips64::cop0_state *cop0;
    mips64::hilo_state *hilo;
    memory::fault fault{};
    bool failed = false;
  };

  using entry_point = std::uint64_t (*)(std::uint64_t *, execution_context *);

  template <std::unsigned_integral T>
  static fn load(execution_context *context, std::uint64_t address) noexcept
      -> std::uint64_t {
    let result = context->memory->load<T>({address});
    if (!result) {
      context->fault = result.error();
      context->failed = true;
      return 0;
    }
    if constexpr (sizeof(T) == 1)
      return static_cast<std::uint64_t>(
          std::int64_t{std::bit_cast<std::int8_t>(*result)});
    if constexpr (sizeof(T) == 4)
      return static_cast<std::uint64_t>(
          std::int64_t{std::bit_cast<std::int32_t>(*result)});
    return *result;
  }

  template <std::unsigned_integral T>
  static fn store(execution_context *context, std::uint64_t address,
                   std::uint64_t value) noexcept -> void {
    let result = context->memory->store<T>({address}, static_cast<T>(value));
    if (!result) {
      context->fault = result.error();
      context->failed = true;
    }
  }

  static fn exception_return(execution_context *context) noexcept
      -> std::uint64_t {
    return context->cop0->exception_return().value;
  }

  static fn syscall(execution_context *context, std::uint64_t pc,
                     std::uint32_t delay_slot) noexcept -> std::uint64_t {
    return context->cop0->syscall({pc}, delay_slot != 0).value;
  }

  static fn write_cop0(execution_context *context, std::uint32_t slot,
                        std::uint64_t value, std::uint32_t wide) noexcept
      -> void {
    if (wide && (slot == 14 || slot == 30)) {
      context->cop0->registers[slot] = value;
      return;
    }
    context->cop0->write(static_cast<std::uint16_t>(slot),
                         static_cast<std::uint32_t>(value));
  }

  compiled_block(std::shared_ptr<asmjit::JitRuntime> runtime, entry_point entry,
                 std::uint16_t slots, bool uses_memory, bool uses_cop0,
                 bool uses_hilo,
                 std::optional<std::uint64_t> branch_pc) noexcept
      : runtime_(std::move(runtime)), entry_(entry), slots_(slots),
        uses_memory_(uses_memory), uses_cop0_(uses_cop0), uses_hilo_(uses_hilo),
        branch_pc_(branch_pc) {}

  std::shared_ptr<asmjit::JitRuntime> runtime_;
  entry_point entry_;
  std::uint16_t slots_;
  bool uses_memory_;
  bool uses_cop0_;
  bool uses_hilo_;
  std::optional<std::uint64_t> branch_pc_;

  fn reset() noexcept -> void {
    // Only this owner can release the installed entry point.
    if (entry_)
      static_cast<void>(runtime_->release(std::exchange(entry_, nullptr)));
  }
};

class compiler {
public:
  [[nodiscard]] static fn create() noexcept -> std::expected<compiler, error> {
    try {
      return compiler{std::make_shared<asmjit::JitRuntime>()};
    } catch (const std::bad_alloc &) {
      return std::unexpected(error{error_code::out_of_memory});
    }
  }

  [[nodiscard]] fn compile(const ir::block &block,
                            std::pmr::memory_resource &scratch) const noexcept
      -> std::expected<compiled_block, error> {
    if (!runtime_ || runtime_->environment().arch() != asmjit::Arch::kX64)
      return std::unexpected(error{error_code::unavailable});
    if (let valid = block.verify(); !valid)
      return std::unexpected(
          error{error_code::invalid_ir, valid.error().node_index});

    const let failure = [](asmjit::Error code, std::size_t index) {
      return std::unexpected(error{code == asmjit::kErrorOutOfMemory
                                       ? error_code::out_of_memory
                                       : error_code::code_generation,
                                   index, code});
    };
    try {
      std::pmr::vector<asmjit::x86::Gp> values{&scratch};
      values.resize(block.nodes().size());
      std::pmr::vector<std::uint8_t> keep_store(block.nodes().size(), 0,
                                                &scratch);
      std::pmr::vector<std::uint8_t> observed(block.info().state_slots, 1,
                                              &scratch);
      const let *conditional = block.branch();
      const bool likely = conditional && conditional->annul_not_taken;
      bool crossed_slot = false;
      for (std::size_t i = block.nodes().size(); i > 0; i--) {
        const let &node = block.nodes()[i - 1];
        if (likely && !crossed_slot &&
            node.source_index <= conditional->source_index) {
          // An annulled slot cannot replace preceding architectural writes.
          std::ranges::fill(observed, 1);
          crossed_slot = true;
        }
        switch (node.op) {
        case ir::opcode::read_state:
          observed[node.slot.index] = 1;
          break;
        case ir::opcode::write_state:
          keep_store[i - 1] = observed[node.slot.index];
          observed[node.slot.index] = 0;
          break;
        case ir::opcode::constant:
        case ir::opcode::shift_left64:
        case ir::opcode::shift_right64:
        case ir::opcode::shift_right_arithmetic64:
        case ir::opcode::shift_right_arithmetic32:
        case ir::opcode::shift_left32:
        case ir::opcode::shift_right32:
        case ir::opcode::sign_extend32:
        case ir::opcode::unsigned_less:
        case ir::opcode::signed_less:
        case ir::opcode::subtract_wrap:
        case ir::opcode::add_wrap:
        case ir::opcode::bit_and:
        case ir::opcode::bit_or:
        case ir::opcode::bit_xor:
          break;
        default:
          // Unclassified effects must observe all preceding state writes.
          std::ranges::fill(observed, 1);
          break;
        }
      }
      error_handler errors;
      asmjit::CodeHolder code;
      code.setErrorHandler(&errors);
      if (const let status =
              code.init(runtime_->environment(), runtime_->cpuFeatures()))
        return failure(status, 0);
      asmjit::x86::Compiler output{&code};
      const let function = output.addFunc(
          asmjit::FuncSignature::build<std::uint64_t, std::uint64_t *,
                                       compiled_block::execution_context *>());
      if (!function)
        return failure(errors.code, 0);
      const let state = output.newUIntPtr();
      if (errors.code)
        return failure(errors.code, 0);
      function->setArg(0, state);
      const let context = output.newUIntPtr();
      if (errors.code)
        return failure(errors.code, 0);
      function->setArg(1, context);
      bool uses_memory = false;
      bool uses_cop0 = false;
      bool uses_hilo = false;

      const let operand = [&](ir::value_id id)
          -> std::expected<asmjit::x86::Gp, asmjit::Error> {
        let &value = values[id.index];
        const let &node = block.nodes()[id.index];
        if (!value.isValid() && node.op == ir::opcode::constant) {
          value = node.type == ir::width::bits64 ? output.newUInt64()
                                                 : output.newUInt32();
          if (errors.code)
            return std::unexpected(errors.code);
          if (const let status =
                  output.mov(value, asmjit::Imm{node.data.immediate}))
            return std::unexpected(status);
        }
        return value;
      };
      const let immediate = [&](ir::value_id id) {
        const let &node = block.nodes()[id.index];
        if (node.op != ir::opcode::constant)
          return false;
        constexpr let positive_limit = static_cast<std::uint64_t>(
            std::numeric_limits<std::int32_t>::max());
        return node.type == ir::width::bits32 ||
               node.data.immediate <= positive_limit ||
               node.data.immediate >=
                   std::numeric_limits<std::uint64_t>::max() - positive_limit;
      };

      const let annulled = output.newLabel();
      if (errors.code)
        return failure(errors.code, block.nodes().size());
      bool guarded_slot = false;
      for (std::size_t i = 0; i < block.nodes().size(); i++) {
        const let &node = block.nodes()[i];
        if (likely && !guarded_slot &&
            node.source_index > conditional->source_index) {
          const let left = operand(conditional->left);
          const let right = operand(conditional->right);
          if (!left || !right)
            return failure(!left ? left.error() : right.error(), i);
          if (const let status = output.cmp(*left, *right))
            return failure(status, i);
          if (const let status = conditional->kind == ir::branch_kind::equal
                                     ? output.jne(annulled)
                                     : output.je(annulled))
            return failure(status, i);
          guarded_slot = true;
        }
        if (node.op == ir::opcode::constant)
          continue;
        if (node.op == ir::opcode::load8_signed ||
            node.op == ir::opcode::load32_signed ||
            node.op == ir::opcode::load64 || node.op == ir::opcode::store8 ||
            node.op == ir::opcode::store32 || node.op == ir::opcode::store64) {
          uses_memory = true;
          const let address = operand(node.data.inputs.left);
          if (!address)
            return failure(address.error(), i);
          asmjit::InvokeNode *call = nullptr;
          asmjit::Error status;
          if (ir::produces_value(node.op)) {
            const let result = output.newUInt64();
            if (errors.code)
              return failure(errors.code, i);
            values[i] = result;
            const let target = node.op == ir::opcode::load8_signed
                                   ? &compiled_block::load<std::uint8_t>
                               : node.op == ir::opcode::load32_signed
                                   ? &compiled_block::load<std::uint32_t>
                                   : &compiled_block::load<std::uint64_t>;
            status = output.invoke(
                &call, asmjit::Imm{target},
                asmjit::FuncSignature::build<
                    std::uint64_t, compiled_block::execution_context *,
                    std::uint64_t>());
            if (status)
              return failure(status, i);
            call->setRet(0, result);
          } else {
            const let value = operand(node.data.inputs.right);
            if (!value)
              return failure(value.error(), i);
            const let target = node.op == ir::opcode::store8
                                   ? &compiled_block::store<std::uint8_t>
                               : node.op == ir::opcode::store32
                                   ? &compiled_block::store<std::uint32_t>
                                   : &compiled_block::store<std::uint64_t>;
            status =
                output.invoke(&call, asmjit::Imm{target},
                              asmjit::FuncSignature::build<
                                  void, compiled_block::execution_context *,
                                  std::uint64_t, std::uint64_t>());
            if (status)
              return failure(status, i);
            call->setArg(2, *value);
          }
          call->setArg(0, context);
          call->setArg(1, *address);
          const let success = output.newLabel();
          if (errors.code)
            return failure(errors.code, i);
          if (const let emitted = output.cmp(
                  asmjit::x86::Mem{
                      context,
                      offsetof(compiled_block::execution_context, failed), 1},
                  0))
            return failure(emitted, i);
          if (const let emitted = output.je(success))
            return failure(emitted, i);
          const let fault_pc = output.newUInt64();
          if (errors.code)
            return failure(errors.code, i);
          if (const let emitted =
                  output.mov(fault_pc, asmjit::Imm{block.source_pc(node)}))
            return failure(emitted, i);
          if (const let emitted = output.ret(fault_pc))
            return failure(emitted, i);
          if (const let emitted = output.bind(success))
            return failure(emitted, i);
          continue;
        }
        if (node.op == ir::opcode::read_hilo ||
            node.op == ir::opcode::write_hilo) {
          uses_hilo = true;
          const let base = output.newUIntPtr();
          if (errors.code)
            return failure(errors.code, i);
          if (const let status = output.mov(
                  base,
                  asmjit::x86::Mem{
                      context,
                      offsetof(compiled_block::execution_context, hilo), 8}))
            return failure(status, i);
          const asmjit::x86::Mem location{base, node.slot.index * 8, 8};
          if (node.op == ir::opcode::read_hilo) {
            const let value = output.newUInt64();
            if (errors.code)
              return failure(errors.code, i);
            if (const let status = output.mov(value, location))
              return failure(status, i);
            values[i] = value;
          } else {
            const let value = operand(node.data.inputs.left);
            if (!value)
              return failure(value.error(), i);
            if (const let status = output.mov(location, *value))
              return failure(status, i);
          }
          continue;
        }
        if (node.op == ir::opcode::exception_return) {
          uses_cop0 = true;
          const let target = output.newUInt64();
          if (errors.code)
            return failure(errors.code, i);
          asmjit::InvokeNode *call = nullptr;
          if (const let status = output.invoke(
                  &call, asmjit::Imm{&compiled_block::exception_return},
                  asmjit::FuncSignature::build<
                      std::uint64_t, compiled_block::execution_context *>()))
            return failure(status, i);
          call->setArg(0, context);
          call->setRet(0, target);
          if (const let status = output.ret(target))
            return failure(status, i);
          continue;
        }
        if (node.op == ir::opcode::syscall) {
          uses_cop0 = true;
          const let vector = output.newUInt64();
          if (errors.code)
            return failure(errors.code, i);
          const let branch = block.branch_pc();
          const bool delay_slot =
              branch && block.source_pc(node) == *branch + 4;
          const let pc = delay_slot ? *branch : block.source_pc(node);
          asmjit::InvokeNode *call = nullptr;
          if (const let status = output.invoke(
                  &call, asmjit::Imm{&compiled_block::syscall},
                  asmjit::FuncSignature::build<
                      std::uint64_t, compiled_block::execution_context *,
                      std::uint64_t, std::uint32_t>()))
            return failure(status, i);
          call->setArg(0, context);
          call->setArg(1, asmjit::Imm{pc});
          call->setArg(2, asmjit::Imm{delay_slot ? 1 : 0});
          call->setRet(0, vector);
          if (const let status = output.ret(vector))
            return failure(status, i);
          continue;
        }
        if (node.op == ir::opcode::read_cop0 ||
            node.op == ir::opcode::read_cop0_64) {
          uses_cop0 = true;
          const let base = output.newUIntPtr();
          const let result = output.newUInt64();
          if (errors.code)
            return failure(errors.code, i);
          if (const let status = output.mov(
                  base,
                  asmjit::x86::Mem{
                      context,
                      offsetof(compiled_block::execution_context, cop0), 8}))
            return failure(status, i);
          const bool wide = node.op == ir::opcode::read_cop0_64;
          const asmjit::x86::Mem location{base, node.slot.index * 8,
                                          wide ? 8u : 4u};
          const let status = wide ? output.mov(result, location)
                                  : output.movsxd(result, location);
          if (status)
            return failure(status, i);
          values[i] = result;
          continue;
        }
        if (node.op == ir::opcode::write_cop0 ||
            node.op == ir::opcode::write_cop0_64) {
          uses_cop0 = true;
          const let value = operand(node.data.inputs.left);
          if (!value)
            return failure(value.error(), i);
          asmjit::InvokeNode *call = nullptr;
          if (const let status = output.invoke(
                  &call, asmjit::Imm{&compiled_block::write_cop0},
                  asmjit::FuncSignature::build<
                      void, compiled_block::execution_context *, std::uint32_t,
                      std::uint64_t, std::uint32_t>()))
            return failure(status, i);
          call->setArg(0, context);
          call->setArg(1, asmjit::Imm{node.slot.index});
          call->setArg(2, *value);
          call->setArg(
              3, asmjit::Imm{node.op == ir::opcode::write_cop0_64 ? 1 : 0});
          continue;
        }
        if (node.op == ir::opcode::write_state) {
          if (!keep_store[i])
            continue;
          const let value = operand(node.data.inputs.left);
          if (!value)
            return failure(value.error(), i);
          const let status = output.mov(
              asmjit::x86::Mem{state, node.slot.index * 8, 8}, *value);
          if (status)
            return failure(status, i);
          continue;
        }
        const let result = node.type == ir::width::bits64 ? output.newUInt64()
                                                          : output.newUInt32();
        if (errors.code)
          return failure(errors.code, i);
        values[i] = result;
        asmjit::Error status = asmjit::kErrorOk;
        switch (node.op) {
        case ir::opcode::constant:
          break;
        case ir::opcode::shift_left64:
        case ir::opcode::shift_right64:
        case ir::opcode::shift_right_arithmetic64: {
          const let input = operand(node.data.shift.value);
          if (!input)
            return failure(input.error(), i);
          status = output.mov(result, *input);
          if (!status)
            status =
                node.op == ir::opcode::shift_left64
                    ? output.shl(result, asmjit::Imm{node.data.shift.amount})
                : node.op == ir::opcode::shift_right64
                    ? output.shr(result, asmjit::Imm{node.data.shift.amount})
                    : output.sar(result, asmjit::Imm{node.data.shift.amount});
          break;
        }
        case ir::opcode::shift_right_arithmetic32:
        case ir::opcode::shift_left32:
        case ir::opcode::shift_right32: {
          const let input = operand(node.data.shift.value);
          if (!input)
            return failure(input.error(), i);
          status = output.mov(result.r32(), input->r32());
          if (!status)
            status = node.op == ir::opcode::shift_left32
                         ? output.shl(result.r32(),
                                      asmjit::Imm{node.data.shift.amount})
                     : node.op == ir::opcode::shift_right32
                         ? output.shr(result.r32(),
                                      asmjit::Imm{node.data.shift.amount})
                         : output.sar(result.r32(),
                                      asmjit::Imm{node.data.shift.amount});
          if (!status)
            status = output.movsxd(result, result.r32());
          break;
        }
        case ir::opcode::sign_extend32: {
          const let input = operand(node.data.inputs.left);
          if (!input)
            return failure(input.error(), i);
          status = output.movsxd(result, input->r32());
          break;
        }
        case ir::opcode::read_state:
          status = output.mov(result,
                              asmjit::x86::Mem{state, node.slot.index * 8, 8});
          break;
        case ir::opcode::unsigned_less:
        case ir::opcode::signed_less: {
          const let left = operand(node.data.inputs.left);
          if (!left)
            return failure(left.error(), i);
          if (const let emitted = output.mov(result, 0))
            return failure(emitted, i);
          const let right = node.data.inputs.right;
          if (immediate(right)) {
            status = output.cmp(
                *left, asmjit::Imm{block.nodes()[right.index].data.immediate});
          } else {
            const let value = operand(right);
            if (!value)
              return failure(value.error(), i);
            status = output.cmp(*left, *value);
          }
          if (!status)
            status = node.op == ir::opcode::unsigned_less
                         ? output.setb(result.r8())
                         : output.setl(result.r8());
          break;
        }
        case ir::opcode::subtract_wrap:
        case ir::opcode::add_wrap:
        case ir::opcode::bit_and:
        case ir::opcode::bit_or:
        case ir::opcode::bit_xor: {
          let left = node.data.inputs.left;
          let right = node.data.inputs.right;
          if (node.op != ir::opcode::subtract_wrap && !immediate(right) &&
              immediate(left))
            std::swap(left, right);
          const let left_value = operand(left);
          if (!left_value)
            return failure(left_value.error(), i);
          status = output.mov(result, *left_value);
          if (status)
            break;
          const let emit = [&](const auto &right_value) {
            if (node.op == ir::opcode::subtract_wrap)
              return output.sub(result, right_value);
            if (node.op == ir::opcode::add_wrap)
              return output.add(result, right_value);
            if (node.op == ir::opcode::bit_and)
              return output.and_(result, right_value);
            if (node.op == ir::opcode::bit_or)
              return output.or_(result, right_value);
            return output.xor_(result, right_value);
          };
          if (immediate(right)) {
            status =
                emit(asmjit::Imm{block.nodes()[right.index].data.immediate});
          } else {
            const let right_value = operand(right);
            if (!right_value)
              return failure(right_value.error(), i);
            status = emit(*right_value);
          }
          break;
        }
        case ir::opcode::read_hilo:
        case ir::opcode::write_hilo:
        case ir::opcode::exception_return:
        case ir::opcode::syscall:
        case ir::opcode::read_cop0_64:
        case ir::opcode::write_cop0_64:
        case ir::opcode::read_cop0:
        case ir::opcode::write_cop0:
        case ir::opcode::write_state:
        case ir::opcode::load8_signed:
        case ir::opcode::load32_signed:
        case ir::opcode::load64:
        case ir::opcode::store8:
        case ir::opcode::store32:
        case ir::opcode::store64:
          break;
        }
        if (status || errors.code)
          return failure(status ? status : errors.code, i);
      }
      if (const let status = output.bind(annulled))
        return failure(status, block.nodes().size());
      const let next_pc = output.newUInt64();
      if (errors.code)
        return failure(errors.code, block.nodes().size());
      if (const let status = output.mov(next_pc, asmjit::Imm{block.next_pc()}))
        return failure(status, block.nodes().size());
      if (block.branch()) {
        const let &branch = *block.branch();
        const let left = operand(branch.left);
        const let right = operand(branch.right);
        if (!left || !right)
          return failure(!left ? left.error() : right.error(),
                         block.nodes().size());
        const let skip = output.newLabel();
        if (errors.code)
          return failure(errors.code, block.nodes().size());
        if (const let status = output.cmp(*left, *right))
          return failure(status, block.nodes().size());
        if (const let status = branch.kind == ir::branch_kind::equal
                                   ? output.jne(skip)
                                   : output.je(skip))
          return failure(status, block.nodes().size());
        if (const let status = output.mov(next_pc, asmjit::Imm{branch.target}))
          return failure(status, block.nodes().size());
        if (const let status = output.bind(skip))
          return failure(status, block.nodes().size());
      }
      if (const let *jump = block.jump())
        if (const let status = output.mov(next_pc, asmjit::Imm{jump->target}))
          return failure(status, block.nodes().size());
      if (const let *jump = block.indirect()) {
        const let target = operand(jump->target);
        if (!target)
          return failure(target.error(), block.nodes().size());
        if (const let status = output.mov(next_pc, *target))
          return failure(status, block.nodes().size());
      }
      if (const let status = output.ret(next_pc))
        return failure(status, block.nodes().size());
      if (const let status = output.endFunc())
        return failure(status, block.nodes().size());
      if (const let status = output.finalize())
        return failure(status, block.nodes().size());
      compiled_block::entry_point entry = nullptr;
      if (const let status = runtime_->add(&entry, &code))
        return failure(status, block.nodes().size());
      return compiled_block{
          runtime_,  entry,     block.info().state_slots, uses_memory,
          uses_cop0, uses_hilo, block.branch_pc()};
    } catch (const std::bad_alloc &) {
      return std::unexpected(error{error_code::out_of_memory});
    }
  }

private:
  struct error_handler : asmjit::ErrorHandler {
    asmjit::Error code = asmjit::kErrorOk;
    fn handleError(asmjit::Error value, const char *,
                     asmjit::BaseEmitter *) noexcept -> void override {
      if (!code)
        code = value;
    }
  };

  explicit compiler(std::shared_ptr<asmjit::JitRuntime> runtime) noexcept
      : runtime_(std::move(runtime)) {}

  std::shared_ptr<asmjit::JitRuntime> runtime_;
};

} // namespace jit::amd64
