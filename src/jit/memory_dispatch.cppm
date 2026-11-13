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
import jit.interpreter;
import jit.exceptions;

export namespace jit {
// One owner. Exact source validation permits writable code and physical
// aliases. A monitored code write or MMIO effect exits before the next
// instruction, after the current instruction's architectural writes commit.
class memory_dispatcher {
public:
  // Keep hot loops resident when unrelated guest PCs share an index.
  static constexpr std::size_t cache_capacity = 8192;
  static constexpr std::size_t cache_ways = 4;
  static constexpr std::size_t cache_sets = cache_capacity / cache_ways;
  static_assert(cache_capacity % cache_ways == 0 &&
                std::has_single_bit(cache_sets));

  [[nodiscard]] static fn create(memory::address_space &memory,
                                  std::uint16_t cold_threshold = 0) noexcept
      -> std::expected<std::unique_ptr<memory_dispatcher>, dispatch_error> {
    let compiler = amd64::compiler::create();
    if (!compiler)
      return std::unexpected(dispatch_error{
          dispatch_error_code::compilation, 0, {}, {}, compiler.error()});
    try {
      return std::unique_ptr<memory_dispatcher>{
          new memory_dispatcher{memory, std::move(*compiler), cold_threshold}};
    } catch (const std::bad_alloc &) {
      return std::unexpected(
          dispatch_error{dispatch_error_code::out_of_memory, 0});
    }
  }

  [[nodiscard]] fn run_slice(mips64::state &state, std::uint64_t budget,
                              std::size_t block_limit = 64,
                              bool dump_ir = false) noexcept
      -> std::expected<dispatch_result, dispatch_error> {
    slice_++;
    if (slice_ == 0) {
      slice_ = 1;
      for (let &set : cache_)
        for (let &entry : set.entries)
          entry.validated_slice = 0;
    }
    dispatch_result result{stop_reason::yielded, 0, 0, 0};
    cached_block *predicted = nullptr;
    std::uint64_t predicted_pc = 0;
    while (result.blocks < block_limit) {
      if (result.instructions == budget) {
        result.reason = stop_reason::budget_exhausted;
        return result;
      }
      if (state.cop0.interrupt_pending()) {
        memory_.invalidate();
        state.pc = state.cop0.exception(0, state.pc, false);
      }
      if (state.pc.value % 4) {
        const let issue =
            exceptions::from_memory({{state.pc.value},
                                     memory::fault_reason::misaligned,
                                     memory::access_kind::execute,
                                     memory::access_size::word},
                                    state.pc.value);
        exceptions::enter(state, memory_, *issue);
        result.instructions++;
        state.cop0.advance_cycles(1);
        result.reason = stop_reason::guest_exception;
        return result;
      }
      const let pc = state.pc.value;
      let &entry = predicted && predicted_pc == pc ? *predicted : select(pc);
      predicted = nullptr;
      if (!valid(entry, pc) ||
          entry.instructions > budget - result.instructions) {
        let compiled =
            compile(entry, pc, dump_ir,
                    std::min<std::uint64_t>(64, budget - result.instructions));
        if (!compiled)
          return fetch_failure(state, compiled.error(), result, budget);
        result.compiled += *compiled;
      }
      if (entry.cold && entry.hits++ >= cold_threshold_ - 1) {
        let warmed =
            compile(entry, pc, dump_ir,
                    std::min<std::uint64_t>(64, budget - result.instructions),
                    false, false);
        if (!warmed)
          return std::unexpected(warmed.error());
        result.compiled += *warmed;
      }
      if (!entry.cold && !entry.tracing_attempted && entry.hits++ >= 2047 &&
          budget - result.instructions >= 64 &&
          block_limit - result.blocks > 1) {
        entry.tracing_attempted = true;
        let promoted = compile(entry, pc, dump_ir, 64, true);
        if (!promoted)
          return std::unexpected(promoted.error());
        result.compiled += *promoted;
      }
      if (entry.instructions > budget - result.instructions) {
        result.reason = stop_reason::budget_exhausted;
        return result;
      }
      const let distance = static_cast<std::uint32_t>(state.cop0.registers[11] -
                                                      state.cop0.registers[9]);
      const let cycle_limit = distance ? std::uint64_t{distance} * 2 -
                                             (state.cop0.half_cycle ? 1u : 0u)
                                       : UINT64_MAX;
      const let executed =
          entry.cold ? interpret(entry, state)
                     : entry.code->execute(
                           state.registers, memory_, &state.cop0, &state.hilo,
                           &state.cop1, entry.source_ranges,
                           budget - result.instructions,
                           static_cast<std::uint32_t>(std::min<std::size_t>(
                               64, block_limit - result.blocks)),
                           cycle_limit);
      if (!executed)
        return execution_failure(state, executed.error(), result);
      state.cop0.advance_cycles(executed->instructions -
                                executed->accounted_instructions);
      state.pc.value = executed->next_pc;
      result.instructions += executed->instructions;
      result.blocks += executed->blocks;
      if (executed->reason == amd64::exit_reason::guest_exception) {
        result.reason = stop_reason::guest_exception;
        return result;
      }
      if (result.blocks < block_limit && result.instructions < budget) {
        predicted_pc = state.pc.value;
        predicted = follow(entry, predicted_pc);
      }
    }
    return result;
  }

private:
  [[gnu::cold, gnu::noinline]] fn
  fetch_failure(mips64::state &state, const dispatch_error &error,
                dispatch_result result, std::uint64_t budget) noexcept
      -> std::expected<dispatch_result, dispatch_error> {
    if (error.code == dispatch_error_code::memory) {
      if (const let issue = exceptions::from_memory(error.access, error.pc,
                                                    error.branch_pc)) {
        if (error.branch_pc) {
          if (budget - result.instructions < 2) {
            result.reason = stop_reason::budget_exhausted;
            return result;
          }
          const let branch_pc = *error.branch_pc;
          const let branch = memory_.fetch<std::uint32_t>({branch_pc});
          if (!branch)
            return std::unexpected(error);
          const std::array<mips64::instruction_word, 2> pair{{{*branch}, {0}}};
          if (!interpreter::supports(pair))
            return std::unexpected(
                dispatch_error{dispatch_error_code::translation, branch_pc});
          const let executed =
              interpreter::execute(pair, {branch_pc}, state, memory_);
          if (!executed)
            return std::unexpected(error);
          state.cop0.advance_cycles(executed->instructions -
                                    executed->accounted_instructions);
          result.instructions += executed->instructions;
          result.blocks++;
          if (executed->instructions == 1) {
            state.pc = {executed->next_pc};
            return result;
          }
        } else {
          state.cop0.advance_cycles(1);
          result.instructions++;
        }
        exceptions::enter(state, memory_, *issue);
        result.reason = stop_reason::guest_exception;
        return result;
      }
      state.pc.value = error.pc;
    }
    return std::unexpected(error);
  }

  [[gnu::cold, gnu::noinline]] fn
  execution_failure(mips64::state &state, const amd64::execution_error &error,
                    dispatch_result result) noexcept
      -> std::expected<dispatch_result, dispatch_error> {
    state.cop0.advance_cycles(error.completed_instructions);
    if (error.code == amd64::error_code::memory_fault)
      if (const let issue = exceptions::from_memory(error.access, error.pc,
                                                    error.branch_pc)) {
        exceptions::enter(state, memory_, *issue);
        state.cop0.advance_cycles(1);
        result.instructions += error.instructions + 1;
        result.blocks += error.blocks;
        result.reason = stop_reason::guest_exception;
        return result;
      }
    state.pc.value = error.pc;
    return std::unexpected(
        dispatch_error{error.code == amd64::error_code::memory_fault
                           ? dispatch_error_code::memory
                           : dispatch_error_code::execution,
                       state.pc.value,
                       error.access,
                       {},
                       {},
                       error.branch_pc,
                       error.code});
  }

  struct cached_block {
    std::optional<amd64::compiled_block> code;
    std::uint64_t pc = 0;
    std::uint64_t generation = 0;
    std::uint64_t validated_slice = 0;
    std::uint64_t external_generation = 0;
    std::array<std::uint64_t, 2> versions{};
    std::array<memory::backing_stamp, 2> stamps{};
    std::uint32_t instructions = 0;
    std::uint16_t hits = 0;
    bool tracing_attempted = false;
    bool cold = false;
    std::array<cached_block *, 2> successors{};
    std::array<mips64::instruction_word, 65> source{};
    std::array<std::span<const std::byte>, 2> source_ranges{};
  };

  struct cache_set {
    std::array<cached_block, cache_ways> entries{};
    std::size_t victim = 0;
  };

  const std::uint16_t cold_threshold_;
  std::uint64_t slice_ = 0;
  memory::address_space &memory_;
  amd64::compiler compiler_;
  std::array<cache_set, cache_sets> cache_{};
  alignas(std::max_align_t) std::array<std::byte, 64 * 1024> scratch_;

  memory_dispatcher(memory::address_space &memory, amd64::compiler compiler,
                    std::uint16_t cold_threshold) noexcept
      : cold_threshold_(cold_threshold), memory_(memory),
        compiler_(std::move(compiler)) {}

  fn interpret(const cached_block &entry, mips64::state &state) noexcept
      -> std::expected<amd64::execution_result, amd64::execution_error> {
    std::array<mips64::instruction_word, 65> words;
    for (std::size_t i = 0; i < entry.instructions; i++) {
      const let bits = entry.source[i].bits;
      words[i] = {std::endian::native == std::endian::little
                      ? std::byteswap(bits)
                      : bits};
    }
    const let result =
        interpreter::execute(std::span{words}.first(entry.instructions),
                             {entry.pc}, state, memory_, entry.source_ranges);
    if (!result)
      return std::unexpected(amd64::execution_error{
          result.error().code == interpreter::error_code::memory
              ? amd64::error_code::memory_fault
              : amd64::error_code::invalid_state,
          result.error().pc, result.error().access, result.error().branch_pc,
          result.error().completed_instructions, result.error().instructions});
    return amd64::execution_result{result->next_pc, result->instructions,
                                   amd64::exit_reason::completed, 1,
                                   result->accounted_instructions};
  }

  fn select(std::uint64_t pc) noexcept -> cached_block & {
    const let index = ((pc >> 2) ^ (pc >> 14) ^ (pc >> 26)) & (cache_sets - 1);
    let &set = cache_[index];
    for (let &entry : set.entries)
      if ((entry.code || entry.cold) && entry.pc == pc)
        return entry;
    for (let &entry : set.entries)
      if (!entry.code && !entry.cold)
        return entry;
    let &entry = set.entries[set.victim];
    set.victim = (set.victim + 1) % cache_ways;
    return entry;
  }

  fn follow(cached_block &entry, std::uint64_t pc) noexcept
      -> cached_block * {
    for (const let successor : entry.successors)
      if (successor && successor->pc == pc &&
          (successor->code || successor->cold))
        return successor;
    let *successor = &select(pc);
    entry.successors[1] = entry.successors[0];
    entry.successors[0] = successor;
    return successor;
  }

  [[gnu::always_inline]] fn valid(cached_block &entry,
                                   std::uint64_t pc) noexcept -> bool {
    if ((!entry.code && !entry.cold) || entry.pc != pc)
      return false;
    if (entry.generation == memory_.mapping_generation() &&
        entry.validated_slice == slice_ &&
        entry.external_generation == memory_.external_generation() &&
        entry.versions[0] == entry.stamps[0].value() &&
        entry.versions[1] == entry.stamps[1].value())
      return true;
    return revalidate(entry, pc);
  }

  [[gnu::noinline]] fn revalidate(cached_block &entry,
                                   std::uint64_t pc) noexcept -> bool {
    if (entry.generation != memory_.mapping_generation()) {
      // Re-resolve before touching borrowed backing after any mapping edit.
      const let first = memory_.executable_tail({pc});
      if (!first || first->data() != entry.source_ranges[0].data())
        return false;
      if (entry.instructions * 4 > first->size()) {
        const let second = memory_.executable_tail({pc + first->size()});
        if (!second || second->data() != entry.source_ranges[1].data())
          return false;
      }
      entry.generation = memory_.mapping_generation();
      entry.validated_slice = 0;
    }
    const let external = memory_.external_generation();
    const std::array versions{entry.stamps[0].value(), entry.stamps[1].value()};
    if (entry.validated_slice == slice_ &&
        entry.external_generation == external && entry.versions == versions)
      return true;
    const let bytes =
        std::as_bytes(std::span{entry.source}).first(entry.instructions * 4);
    const let first = std::min(
        bytes.size(), memory::page_size - (pc & (memory::page_size - 1)));
    const bool unchanged =
        std::memcmp(entry.source_ranges[0].data(), bytes.data(), first) == 0 &&
        (first == bytes.size() ||
         std::memcmp(entry.source_ranges[1].data(), bytes.data() + first,
                     bytes.size() - first) == 0);
    if (unchanged) {
      entry.validated_slice = slice_;
      entry.external_generation = external;
      entry.versions = versions;
    }
    return unchanged;
  }

  fn compile(cached_block &entry, std::uint64_t pc, bool dump_ir,
               std::uint64_t limit, bool trace = false,
               bool allow_cold = true) noexcept
      -> std::expected<bool, dispatch_error> {
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
      if (delay) {
        delay = false;
        if (!trace)
          break;
        continue;
      }
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
          op == 0x2f || op == 0x10)
        if (!trace)
          break;
    }
    const bool cold = allow_cold && cold_threshold_ && !trace && !dump_ir &&
                      interpreter::supports(std::span{words}.first(count));
    std::expected<native_block, dispatch_error> native =
        std::unexpected(dispatch_error{dispatch_error_code::translation, pc});
    if (trace) {
      let promoted = compile_native_trace(std::span{words}.first(count), {pc},
                                           compiler_, scratch_);
      if (!promoted)
        return std::unexpected(promoted.error());
      if (!*promoted)
        return false;
      native = std::move(**promoted);
    } else if (!cold) {
      native = compile_native_block(std::span{words}.first(count), {pc},
                                    compiler_, scratch_, dump_ir, true);
    }
    if (!cold && !native &&
        native.error().code == dispatch_error_code::translation) {
      let prefix = native.error().translation.instruction_index;
      if (prefix && mips64::has_delay_slot(words[prefix - 1]))
        prefix--;
      if (prefix)
        native = compile_native_block(std::span{words}.first(prefix), {pc},
                                      compiler_, scratch_, dump_ir, true);
    }
    if (!cold && !native)
      return std::unexpected(native.error());
    const let instructions =
        cold ? static_cast<std::uint32_t>(count) : native->instructions;
    const let first = memory_.executable_tail({pc});
    if (!first)
      return std::unexpected(
          dispatch_error{dispatch_error_code::memory, pc, first.error()});
    std::array<const std::byte *, 2> host{first->data(), nullptr};
    if (instructions * 4 > first->size()) {
      const let second = memory_.executable_tail({pc + first->size()});
      if (!second)
        return std::unexpected(dispatch_error{
            dispatch_error_code::memory, pc + first->size(), second.error()});
      host[1] = second->data();
    }
    const let bytes = instructions * 4;
    const let first_bytes = std::min<std::size_t>(bytes, first->size());
    entry.source_ranges = {std::span{host[0], first_bytes},
                           std::span{host[1], bytes - first_bytes}};
    entry.stamps = {memory_.stamp(entry.source_ranges[0]),
                    memory_.stamp(entry.source_ranges[1])};
    entry.pc = pc;
    entry.validated_slice = 0;
    entry.hits = 0;
    entry.tracing_attempted = trace;
    entry.generation = memory_.mapping_generation();
    entry.instructions = instructions;
    entry.cold = cold;
    for (std::size_t i = 0; i < instructions; i++) {
      const let bits = words[i].bits;
      entry.source[i].bits = std::endian::native == std::endian::little
                                 ? std::byteswap(bits)
                                 : bits;
    }
    entry.code.reset();
    if (!cold)
      entry.code.emplace(std::move(native->code));
    return !cold;
  }
};
} // namespace jit
