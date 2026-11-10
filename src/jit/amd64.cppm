module;

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <memory_resource>
#include <new>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <dynamips/macros.h>

export module jit.amd64;

import asmjit;
import jit.ir;

export namespace jit::amd64 {

enum class error_code {
  invalid_ir,
  out_of_memory,
  code_generation,
  invalid_state,
  unavailable,
};

struct error {
  error_code code;
  std::size_t node_index = 0;
  std::uint32_t native_code = 0;
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
        entry_(std::exchange(other.entry_, nullptr)), slots_(other.slots_) {}

  fn operator=(compiled_block &&other) noexcept -> compiled_block & {
    if (this != &other) {
      reset();
      runtime_ = std::move(other.runtime_);
      entry_ = std::exchange(other.entry_, nullptr);
      slots_ = other.slots_;
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
    return entry_(state.data());
  }

private:
  friend class compiler;
  using entry_point = std::uint64_t (*)(std::uint64_t *);

  compiled_block(std::shared_ptr<asmjit::JitRuntime> runtime, entry_point entry,
                 std::uint16_t slots) noexcept
      : runtime_(std::move(runtime)), entry_(entry), slots_(slots) {}

  std::shared_ptr<asmjit::JitRuntime> runtime_;
  entry_point entry_;
  std::uint16_t slots_;

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
      error_handler errors;
      asmjit::CodeHolder code;
      code.setErrorHandler(&errors);
      if (const let status =
              code.init(runtime_->environment(), runtime_->cpuFeatures()))
        return failure(status, 0);
      asmjit::x86::Compiler output{&code};
      const let function = output.addFunc(
          asmjit::FuncSignature::build<std::uint64_t, std::uint64_t *>());
      if (!function)
        return failure(errors.code, 0);
      const let state = output.newUIntPtr();
      if (errors.code)
        return failure(errors.code, 0);
      function->setArg(0, state);

      for (std::size_t i = 0; i < block.nodes().size(); i++) {
        const let &node = block.nodes()[i];
        if (node.op == ir::opcode::write_state) {
          const let status =
              output.mov(asmjit::x86::Mem{state, node.slot.index * 8, 8},
                         values[node.data.inputs.left.index]);
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
          status = output.mov(result, asmjit::Imm{node.data.immediate});
          break;
        case ir::opcode::read_state:
          status = output.mov(result,
                              asmjit::x86::Mem{state, node.slot.index * 8, 8});
          break;
        case ir::opcode::add_wrap:
        case ir::opcode::bit_or:
        case ir::opcode::bit_xor: {
          status = output.mov(result, values[node.data.inputs.left.index]);
          if (status)
            break;
          const let right = values[node.data.inputs.right.index];
          if (node.op == ir::opcode::add_wrap)
            status = output.add(result, right);
          else if (node.op == ir::opcode::bit_or)
            status = output.or_(result, right);
          else
            status = output.xor_(result, right);
          break;
        }
        case ir::opcode::write_state:
          break;
        }
        if (status || errors.code)
          return failure(status ? status : errors.code, i);
      }
      const let next_pc = output.newUInt64();
      if (errors.code)
        return failure(errors.code, block.nodes().size());
      if (const let status = output.mov(next_pc, asmjit::Imm{block.next_pc()}))
        return failure(status, block.nodes().size());
      if (const let status = output.ret(next_pc))
        return failure(status, block.nodes().size());
      if (const let status = output.endFunc())
        return failure(status, block.nodes().size());
      if (const let status = output.finalize())
        return failure(status, block.nodes().size());
      compiled_block::entry_point entry = nullptr;
      if (const let status = runtime_->add(&entry, &code))
        return failure(status, block.nodes().size());
      return compiled_block{runtime_, entry, block.info().state_slots};
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
