module;
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
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
  // Keep hot loops resident when unrelated guest PCs share an index.
  static constexpr std::size_t cache_capacity = 8192;
  static constexpr std::size_t cache_ways = 4;
  static constexpr std::size_t cache_sets = cache_capacity / cache_ways;
  static_assert(cache_capacity % cache_ways == 0 &&
                std::has_single_bit(cache_sets));

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
      if (state.cop0.interrupt_pending()) {
        memory_.clear_reservation();
        state.pc = state.cop0.exception(0, state.pc, false);
      }
      if (state.pc.value % 4)
        return std::unexpected(
            dispatch_error{dispatch_error_code::invalid_pc, state.pc.value});
      const let pc = state.pc.value;
      let &entry = select(pc);
      if (!valid(entry, pc) ||
          entry.instructions > budget - result.instructions) {
        if (let compiled = compile(
                entry, pc, dump_ir,
                std::min<std::uint64_t>(64, budget - result.instructions));
            !compiled) {
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
      const let executed = entry.code->execute(
          state.registers, memory_, &state.cop0, &state.hilo, &state.cop1);
      if (!executed) {
        state.pc.value = executed.error().pc;
        return std::unexpected(dispatch_error{
            executed.error().code == amd64::error_code::memory_fault
                ? dispatch_error_code::memory
                : dispatch_error_code::execution,
            state.pc.value,
            executed.error().access,
            {},
            {},
            executed.error().branch_pc,
            executed.error().code});
      }
      state.cop0.advance_cycles(executed->instructions);
      state.pc.value = executed->next_pc;
      result.instructions += executed->instructions;
      result.blocks++;
      if (executed->reason == amd64::exit_reason::guest_exception) {
        result.reason = stop_reason::guest_exception;
        return result;
      }
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
    std::array<const std::byte *, 2> host{};
  };

  struct cache_set {
    std::array<cached_block, cache_ways> entries{};
    std::size_t victim = 0;
  };

  memory::address_space &memory_;
  amd64::compiler compiler_;
  std::array<cache_set, cache_sets> cache_{};
  alignas(std::max_align_t) std::array<std::byte, 64 * 1024> scratch_;

  memory_dispatcher(memory::address_space &memory,
                    amd64::compiler compiler) noexcept
      : memory_(memory), compiler_(std::move(compiler)) {}

  fn select(std::uint64_t pc) noexcept -> cached_block & {
    const let index = ((pc >> 2) ^ (pc >> 14) ^ (pc >> 26)) & (cache_sets - 1);
    let &set = cache_[index];
    for (let &entry : set.entries)
      if (entry.code && entry.pc == pc)
        return entry;
    for (let &entry : set.entries)
      if (!entry.code)
        return entry;
    let &entry = set.entries[set.victim];
    set.victim = (set.victim + 1) % cache_ways;
    return entry;
  }

  fn valid(cached_block &entry, std::uint64_t pc) noexcept -> bool {
    if (!entry.code || entry.pc != pc)
      return false;
    if (entry.generation != memory_.mapping_generation()) {
      // Re-resolve before touching borrowed backing after any mapping edit.
      const let first = memory_.executable_tail({pc});
      if (!first || first->data() != entry.host[0])
        return false;
      if (entry.instructions * 4 > first->size()) {
        const let second = memory_.executable_tail({pc + first->size()});
        if (!second || second->data() != entry.host[1])
          return false;
      }
      entry.generation = memory_.mapping_generation();
    }
    const let bytes =
        std::as_bytes(std::span{entry.source}).first(entry.instructions * 4);
    const let first = std::min(
        bytes.size(), memory::page_size - (pc & (memory::page_size - 1)));
    return std::memcmp(entry.host[0], bytes.data(), first) == 0 &&
           (first == bytes.size() ||
            std::memcmp(entry.host[1], bytes.data() + first,
                        bytes.size() - first) == 0);
  }

  fn compile(cached_block &entry, std::uint64_t pc, bool dump_ir,
               std::uint64_t limit) noexcept
      -> std::expected<void, dispatch_error> {
    std::array<mips64::instruction_word, 65> words{};
    std::size_t count = 0;
    bool delay = false;
    while (count < limit || delay) {
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
        if (count == limit && count > 1) {
          count--;
          break;
        }
        delay = true;
        continue;
      }
      const let op = *fetched >> 26;
      if ((op == 0 && ((*fetched & 63) == 0x0c || (*fetched & 63) == 0x0f)) ||
          (op >= 0x28 && op <= 0x2f) || op >= 0x38 || op == 0x10)
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
    const let first = memory_.executable_tail({pc});
    if (!first)
      return std::unexpected(
          dispatch_error{dispatch_error_code::memory, pc, first.error()});
    std::array<const std::byte *, 2> host{first->data(), nullptr};
    if (native->instructions * 4 > first->size()) {
      const let second = memory_.executable_tail({pc + first->size()});
      if (!second)
        return std::unexpected(dispatch_error{
            dispatch_error_code::memory, pc + first->size(), second.error()});
      host[1] = second->data();
    }
    entry.host = host;
    entry.pc = pc;
    entry.generation = memory_.mapping_generation();
    entry.instructions = native->instructions;
    for (std::size_t i = 0; i < native->instructions; i++) {
      const let bits = words[i].bits;
      entry.source[i].bits = std::endian::native == std::endian::little
                                 ? std::byteswap(bits)
                                 : bits;
    }
    entry.code.emplace(std::move(native->code));
    return {};
  }
};
} // namespace jit
