module;
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <utility>
#include <dynamips/macros.h>
export module jit.memory_dispatch;
import jit.dispatch;
import jit.mips64;
import jit.memory;
import jit.amd64;

export namespace jit {
// One owner. Exact source validation permits writable code and physical
// aliases. Stores end blocks so later instructions are fetched after their
// effects.
class memory_dispatcher {
public:
  static constexpr std::size_t cache_capacity = 1024;

  [[nodiscard]] static fn create(memory::address_space &memory) noexcept
      -> std::expected<std::unique_ptr<memory_dispatcher>, dispatch_error> {
    let compiler = amd64::compiler::create();
    if (!compiler)
      return std::unexpected(dispatch_error{
          dispatch_error_code::compilation, 0, {}, {}, compiler.error()});
    try {
      return std::unique_ptr<memory_dispatcher>{
          new memory_dispatcher{memory, std::move(*compiler)}};
    } catch (const std::bad_alloc &) {
      return std::unexpected(
          dispatch_error{dispatch_error_code::out_of_memory, 0});
    }
  }

  [[nodiscard]] fn run_slice(mips64::state &state, std::uint64_t budget,
                              std::size_t block_limit = 64,
                              bool dump_ir = false) noexcept
      -> std::expected<dispatch_result, dispatch_error> {
    dispatch_result result{stop_reason::yielded, 0, 0, 0};
    while (result.blocks < block_limit) {
      if (result.instructions == budget) {
        result.reason = stop_reason::budget_exhausted;
        return result;
      }
      if (state.pc.value % 4)
        return std::unexpected(
            dispatch_error{dispatch_error_code::invalid_pc, state.pc.value});
      const let pc = state.pc.value;
      const let index =
          ((pc >> 2) ^ (pc >> 14) ^ (pc >> 26)) & (cache_capacity - 1);
      let &entry = cache_[index];
      if (!valid(entry, pc)) {
        if (let compiled = compile(entry, pc, dump_ir); !compiled) {
          if (compiled.error().code == dispatch_error_code::memory)
            state.pc.value = compiled.error().pc;
          return std::unexpected(compiled.error());
        }
        result.compiled++;
      }
      if (entry.instructions > budget - result.instructions) {
        result.reason = stop_reason::budget_exhausted;
        return result;
      }
      const let executed =
          entry.code->run(state.registers, memory_, &state.cop0, &state.hilo);
      if (!executed) {
        state.pc.value = executed.error().pc;
        return std::unexpected(dispatch_error{dispatch_error_code::memory,
                                              state.pc.value,
                                              executed.error().access,
                                              {},
                                              {},
                                              executed.error().branch_pc});
      }
      state.pc.value = *executed;
      result.instructions += entry.instructions;
      result.blocks++;
    }
    return result;
  }

private:
  struct cached_block {
    std::optional<amd64::compiled_block> code;
    std::uint64_t pc = 0;
    std::uint64_t generation = 0;
    std::uint32_t instructions = 0;
    std::array<mips64::instruction_word, 65> source{};
  };

  memory::address_space &memory_;
  amd64::compiler compiler_;
  std::array<cached_block, cache_capacity> cache_{};
  alignas(std::max_align_t) std::array<std::byte, 64 * 1024> scratch_;

  memory_dispatcher(memory::address_space &memory,
                    amd64::compiler compiler) noexcept
      : memory_(memory), compiler_(std::move(compiler)) {}

  fn valid(const cached_block &entry, std::uint64_t pc) noexcept -> bool {
    if (!entry.code || entry.pc != pc ||
        entry.generation != memory_.mapping_generation())
      return false;
    for (std::size_t i = 0; i < entry.instructions; i++) {
      const let word = memory_.fetch<std::uint32_t>({pc + i * 4});
      if (!word || *word != entry.source[i].bits)
        return false;
    }
    return true;
  }

  fn compile(cached_block &entry, std::uint64_t pc, bool dump_ir) noexcept
      -> std::expected<void, dispatch_error> {
    std::array<mips64::instruction_word, 65> words{};
    std::size_t count = 0;
    bool delay = false;
    while (count < 64 || delay) {
      if (count > (UINT64_MAX - pc) / 4)
        return std::unexpected(
            dispatch_error{dispatch_error_code::invalid_pc, pc});
      const let fetched = memory_.fetch<std::uint32_t>({pc + count * 4});
      if (!fetched) {
        const let prefix = delay ? count - 1 : count;
        if (prefix != 0) {
          count = prefix;
          break;
        }
        return std::unexpected(
            dispatch_error{dispatch_error_code::memory,
                           pc + count * 4,
                           fetched.error(),
                           {},
                           {},
                           delay ? std::optional{pc} : std::nullopt});
      }
      words[count] = {*fetched};
      count++;
      if (delay)
        break;
      if (mips64::has_delay_slot({*fetched})) {
        delay = true;
        continue;
      }
      const let op = *fetched >> 26;
      if ((op == 0 && (*fetched & 63) == 0x0c) || (op >= 0x28 && op <= 0x2f) ||
          op >= 0x38 || op == 0x10)
        break;
    }
    let native = compile_native_block(std::span{words}.first(count), {pc},
                                       compiler_, scratch_, dump_ir);
    if (!native && native.error().code == dispatch_error_code::translation) {
      let prefix = native.error().translation.instruction_index;
      if (prefix && mips64::has_delay_slot(words[prefix - 1]))
        prefix--;
      if (prefix)
        native = compile_native_block(std::span{words}.first(prefix), {pc},
                                      compiler_, scratch_, dump_ir);
    }
    if (!native)
      return std::unexpected(native.error());
    entry.pc = pc;
    entry.generation = memory_.mapping_generation();
    entry.instructions = native->instructions;
    entry.source = words;
    entry.code.emplace(std::move(native->code));
    return {};
  }
};
} // namespace jit
