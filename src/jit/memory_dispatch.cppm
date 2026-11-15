module;
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <memory_resource>
#include <vector>
#include <new>
#include <optional>
#include <span>
#include <utility>
#include <dynamips/macros.h>
export module jit.memory_dispatch;
import jit.dispatch;
import jit.ir;
import jit.mips64;
import jit.memory;
import jit.amd64;
import jit.interpreter;
import jit.exceptions;
import jit.count_batch;

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

  [[nodiscard]] fn region_compilations() const noexcept -> std::size_t {
    return region_compilations_;
  }

  [[nodiscard]] fn largest_region() const noexcept -> std::size_t {
    return largest_region_;
  }

  [[nodiscard]] fn region_block_count(std::uint64_t pc) noexcept
      -> std::size_t {
    const let *entry = find_cached(pc);
    return entry && entry->region ? entry->region->blocks : 0;
  }

  [[nodiscard]] fn region_closed_to_root(std::uint64_t pc) noexcept -> bool {
    const let *entry = find_cached(pc);
    return entry && entry->region && entry->region->closed_to_root;
  }

  [[nodiscard]] fn region_backedge(std::uint64_t pc) noexcept
      -> std::optional<std::size_t> {
    const let *entry = find_cached(pc);
    return entry && entry->region ? entry->region->backedge : std::nullopt;
  }

  [[nodiscard]] fn region_watch_count(std::uint64_t pc) noexcept
      -> std::size_t {
    const let *entry = find_cached(pc);
    return entry && entry->region ? entry->region->range_count : 0;
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
    amd64::compiled_block::execution_session native{state, memory_};
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
      if (entry.region && entry.region->disjoint_state != &state) {
        if (!source_disjoint(std::span{entry.region->ranges}.first(
                                 entry.region->range_count),
                             state))
          entry = {};
        else
          entry.region->disjoint_state = &state;
      }
      if (!valid(entry, pc) ||
          entry.instructions > budget - result.instructions) {
        let compiled =
            compile(entry, state, pc, dump_ir,
                    std::min<std::uint64_t>(64, budget - result.instructions));
        if (!compiled)
          return fetch_failure(state, compiled.error(), result, budget);
        result.compiled += *compiled;
      }
      if (entry.count_pattern && source_disjoint(entry.source_ranges, state) &&
          count_tail_complete(pc)) {
        if (const let batch = count_batch::run(*entry.count_pattern, state,
                                               budget - result.instructions,
                                               block_limit - result.blocks)) {
          result.instructions += batch->instructions;
          result.blocks += batch->blocks;
          continue;
        }
      }
      if (entry.cold && entry.hits++ >= cold_threshold_ - 1) {
        let warmed =
            compile(entry, state, pc, dump_ir,
                    std::min<std::uint64_t>(64, budget - result.instructions),
                    false, false);
        if (!warmed)
          return std::unexpected(warmed.error());
        result.compiled += *warmed;
      }
      if (!entry.cold && !entry.tracing_attempted &&
          memory_.may_speculate_translation() && entry.hits++ >= 2047 &&
          budget - result.instructions >= 64 &&
          block_limit - result.blocks > 1) {
        entry.tracing_attempted = true;
        let promoted = compile(entry, state, pc, dump_ir, 64, true);
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
          entry.cold
              ? interpret(entry, state)
              : entry.code->execute(
                    native, code_ranges(entry), budget - result.instructions,
                    static_cast<std::uint32_t>(
                        std::min<std::size_t>(64, block_limit - result.blocks)),
                    cycle_limit, memory::range_stability::stable);
      if (!executed)
        return execution_failure(state, executed.error(), result);
      if (!entry.fused && executed->reason == amd64::exit_reason::completed &&
          executed->instructions == entry.instructions) {
        entry.first_successor_pc = executed->next_pc;
        entry.first_words = entry.instructions;
        entry.successor_observed = true;
      }
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

  struct region_identity {
    struct fragment {
      std::uint64_t pc;
      std::size_t source_offset;
      std::span<const std::byte> backing;
    };

    std::array<std::byte, 256 * 4> source{};
    std::array<fragment, 64> fragments{};

    struct watch_page {
      const std::byte *base = nullptr;
      std::size_t begin = 0, end = 0;
    };

    std::array<watch_page, 4> watch_pages{};
    std::array<std::span<const std::byte>, 4> ranges{};
    std::array<std::uint64_t, 4> pages{};
    std::array<memory::backing_stamp, 4> stamps{};
    std::array<std::uint64_t, 4> versions{};
    std::size_t fragment_count = 0, page_count = 0, bytes = 0, blocks = 0;
    std::size_t range_count = 0;
    bool closed_to_root = false;
    const mips64::state *disjoint_state = nullptr;
    std::optional<std::size_t> backedge{};
  };

  struct cached_block {
    std::unique_ptr<region_identity> region;
    std::optional<amd64::compiled_block> code;
    std::uint64_t pc = 0;
    std::uint64_t generation = 0;
    std::uint64_t validated_slice = 0;
    std::uint64_t external_generation = 0;
    std::array<std::uint64_t, 2> versions{};
    std::array<memory::backing_stamp, 2> stamps{};
    std::uint32_t instructions = 0;
    std::uint32_t source_words = 0;
    std::optional<count_batch::loop> count_pattern{};
    std::uint16_t hits = 0;
    bool tracing_attempted = false;
    bool cold = false;
    bool speculative_source = false;
    bool fused = false, successor_observed = false;
    std::uint64_t first_successor_pc = 0;
    std::uint32_t first_words = 0;
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
  std::size_t region_compilations_ = 0, largest_region_ = 0;
  memory::address_space &memory_;
  amd64::compiler compiler_;
  std::array<cache_set, cache_sets> cache_{};
  alignas(std::max_align_t) std::array<std::byte, 64 * 1024> scratch_;
  alignas(std::max_align_t) std::array<std::byte, 512 * 1024> region_scratch_;

  memory_dispatcher(memory::address_space &memory, amd64::compiler compiler,
                    std::uint16_t cold_threshold) noexcept
      : cold_threshold_(cold_threshold), memory_(memory),
        compiler_(std::move(compiler)) {}

  [[gnu::cold, gnu::noinline]] fn interpret(const cached_block &entry,
                                             mips64::state &state) noexcept
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

  [[gnu::noinline]] fn select(std::uint64_t pc) noexcept -> cached_block & {
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

  [[gnu::always_inline]] inline fn follow(cached_block &entry,
                                           std::uint64_t pc) noexcept
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

  // A prior short slice can leave the four-slot tail split into more blocks.
  [[nodiscard]] fn count_tail_complete(std::uint64_t pc) const noexcept
      -> bool {
    const let tail = pc + 4;
    const let index =
        ((tail >> 2) ^ (tail >> 14) ^ (tail >> 26)) & (cache_sets - 1);
    for (const let &entry : cache_[index].entries)
      if ((entry.code || entry.cold) && entry.pc == tail)
        return entry.instructions >= 4;
    return true;
  }

  [[nodiscard]] static inline fn
  source_disjoint(std::span<const std::span<const std::byte>> ranges,
                  const mips64::state &state) noexcept -> bool {
    const let begin = reinterpret_cast<std::uintptr_t>(&state);
    if (sizeof(state) > UINTPTR_MAX - begin)
      return false;
    const let end = begin + sizeof(state);
    for (const let range : ranges) {
      if (range.empty())
        continue;
      const let source = reinterpret_cast<std::uintptr_t>(range.data());
      if (range.size() > UINTPTR_MAX - source ||
          (source < end && begin < source + range.size()))
        return false;
    }
    return true;
  }

  [[nodiscard]] inline fn validated(const cached_block &entry) const noexcept
      -> bool {
    return entry.validated_slice == slice_ ||
           (!memory_.requires_slice_validation() && entry.validated_slice != 0);
  }

  [[gnu::always_inline]] fn valid(cached_block &entry,
                                   std::uint64_t pc) noexcept -> bool {
    if ((!entry.code && !entry.cold) || entry.pc != pc)
      return false;
    if (entry.speculative_source && !memory_.may_speculate_translation())
      return false;
    if (entry.region)
      return valid_region(entry);
    if (entry.generation == memory_.mapping_generation() && validated(entry) &&
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
      if (entry.source_words * 4 > first->size()) {
        const let second = memory_.executable_tail({pc + first->size()});
        if (!second || second->data() != entry.source_ranges[1].data())
          return false;
      }
      entry.generation = memory_.mapping_generation();
      entry.validated_slice = 0;
    }
    const let external = memory_.external_generation();
    const std::array versions{entry.stamps[0].value(), entry.stamps[1].value()};
    if (validated(entry) && entry.external_generation == external &&
        entry.versions == versions)
      return true;
    const let bytes =
        std::as_bytes(std::span{entry.source}).first(entry.source_words * 4);
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

  fn code_ranges(const cached_block &entry) const noexcept
      -> std::span<const std::span<const std::byte>> {
    if (entry.region)
      return std::span{entry.region->ranges}.first(entry.region->range_count);
    return entry.source_ranges;
  }

  [[gnu::noinline]] fn valid_region(cached_block &entry) noexcept -> bool {
    let &identity = *entry.region;
    if (entry.generation != memory_.mapping_generation()) {
      // Resolve every fragment before reading any borrowed backing or stamp.
      for (std::size_t i = 0; i < identity.fragment_count; i++) {
        const let &fragment = identity.fragments[i];
        const let current = memory_.executable_tail({fragment.pc});
        if (!current || current->size() < fragment.backing.size() ||
            current->data() != fragment.backing.data())
          return false;
      }
      entry.generation = memory_.mapping_generation();
      entry.validated_slice = 0;
    }
    bool unchanged = validated(entry) &&
                     entry.external_generation == memory_.external_generation();
    for (std::size_t i = 0; i < identity.page_count; i++)
      unchanged =
          unchanged && identity.versions[i] == identity.stamps[i].value();
    if (unchanged)
      return true;
    for (std::size_t i = 0; i < identity.fragment_count; i++) {
      const let &fragment = identity.fragments[i];
      if (std::memcmp(fragment.backing.data(),
                      identity.source.data() + fragment.source_offset,
                      fragment.backing.size()) != 0)
        return false;
    }
    for (std::size_t i = 0; i < identity.page_count; i++)
      identity.versions[i] = identity.stamps[i].value();
    entry.validated_slice = slice_;
    entry.external_generation = memory_.external_generation();
    return true;
  }

  fn find_cached(std::uint64_t pc) noexcept -> cached_block * {
    const let index = ((pc >> 2) ^ (pc >> 14) ^ (pc >> 26)) & (cache_sets - 1);
    for (let &candidate : cache_[index].entries)
      if ((candidate.code || candidate.cold) && candidate.pc == pc)
        return &candidate;
    return nullptr;
  }

  // Predictions choose compilation candidates only. Native edges still check
  // the architectural branch condition or indirect target before transferring.
  fn compile_region(cached_block &entry, const mips64::state &state,
                      std::uint64_t pc) noexcept
      -> std::expected<bool, dispatch_error> {

    if (!memory_.may_speculate_translation())
      return false;
    try {
      std::pmr::monotonic_buffer_resource arena{
          region_scratch_.data(), region_scratch_.size(),
          std::pmr::null_memory_resource()};
      std::pmr::vector<ir::block> blocks{&arena};
      blocks.reserve(32);
      let identity = std::make_unique<region_identity>();
      std::size_t words_used = 0;
      std::array<std::uint64_t, 8> returns{};
      std::size_t return_depth = 0;

      struct path_context {
        std::uint64_t pc = 0;
        std::size_t depth = 0;
        std::array<std::uint64_t, 8> returns{};
      };

      std::array<path_context, 32> contexts{};
      let current_pc = pc;
      while (blocks.size() < 32 && words_used < 256) {
        std::optional<std::size_t> repeated;
        for (std::size_t i = 0; i < blocks.size(); i++) {
          const let &prior = contexts[i];
          if (prior.pc == current_pc && prior.depth == return_depth &&
              std::equal(returns.begin(), returns.begin() + return_depth,
                         prior.returns.begin())) {
            repeated = i;
            break;
          }
        }
        if (repeated) {
          identity->closed_to_root = *repeated == 0;
          identity->backedge = repeated;
          break;
        }
        contexts[blocks.size()] = {current_pc, return_depth, returns};
        std::array<mips64::instruction_word, 65> words{};
        std::size_t count = 0;
        bool delay = false, barrier = false;
        const let maximum = std::min<std::size_t>(64, 256 - words_used);
        while (count < maximum || delay) {
          if (count >= words.size() || count >= 256 - words_used ||
              count > (UINT64_MAX - current_pc) / 4) {
            if (delay && count)
              count--;
            break;
          }
          // Speculative candidates must be ordinary executable backing. Never
          // invoke a device callback while discovering an untaken path.
          const let backing = memory_.executable_tail({current_pc + count * 4});
          if (!backing || backing->size() < sizeof(std::uint32_t) ||
              !backing->data()) {
            if (delay && count)
              count--;
            break;
          }
          std::uint32_t fetched;
          std::memcpy(&fetched, backing->data(), sizeof(fetched));
          if constexpr (std::endian::native == std::endian::little)
            fetched = std::byteswap(fetched);
          const let op = fetched >> 26;
          // Keep CP0, CACHE and explicit serialization outside native regions.
          barrier =
              op == 0x10 || op == 0x2f ||
              (op == 0 && ((fetched & 63) == 0x0c || (fetched & 63) == 0x0d ||
                           (fetched & 63) == 0x0f));
          if (barrier)
            break;
          words[count++] = {fetched};
          if (delay)
            break;
          delay = mips64::has_delay_slot({fetched});
        }
        if (!count || barrier)
          break;
        let translated = mips64::translate(std::span{words}.first(count),
                                            {current_pc}, arena);
        if (!translated) {
          let prefix = translated.error().instruction_index;
          if (prefix && mips64::has_delay_slot(words[prefix - 1]))
            prefix--;
          if (!prefix)
            break;
          translated = mips64::translate(std::span{words}.first(prefix),
                                         {current_pc}, arena);
        }
        if (!translated)
          break;
        count = translated->info().instruction_count;
        if (count > (UINT64_MAX - current_pc) / 4)
          break;
        std::array<region_identity::fragment, 2> fragments{};
        std::size_t fragment_count = 0, copied = 0;
        bool unavailable = false;
        std::array<std::uint64_t, 2> added_pages{};
        std::size_t added_count = 0;
        while (copied < count * 4) {
          const let at = current_pc + copied;
          const let backing = memory_.executable_tail({at});
          if (!backing) {
            unavailable = true;
            break;
          }
          const let size =
              std::min<std::size_t>(count * 4 - copied, backing->size());
          if (!size || fragment_count == fragments.size()) {
            unavailable = true;
            break;
          }
          fragments[fragment_count++] = {at, identity->bytes + copied,
                                         backing->first(size)};
          const let page = at & ~std::uint64_t{memory::page_size - 1};
          bool known = false;
          for (std::size_t i = 0; i < identity->page_count; i++)
            known = known || identity->pages[i] == page;
          if (!known)
            added_pages[added_count++] = page;
          copied += size;
        }
        if (unavailable || identity->page_count + added_count > 4 ||
            identity->fragment_count + fragment_count >
                identity->fragments.size())
          break;
        let watches = identity->watch_pages;
        let watch_count = identity->range_count;
        for (std::size_t i = 0; i < fragment_count; i++) {
          const let &fragment = fragments[i];
          const let offset =
              static_cast<std::size_t>(fragment.pc & (memory::page_size - 1));
          const let *base = fragment.backing.data() - offset;
          std::size_t watch = 0;
          while (watch < watch_count && watches[watch].base != base)
            watch++;
          if (watch == watch_count) {
            if (watch_count == watches.size()) {
              unavailable = true;
              break;
            }
            watches[watch_count++] = {base, offset,
                                      offset + fragment.backing.size()};
          } else {
            watches[watch].begin = std::min(watches[watch].begin, offset);
            watches[watch].end =
                std::max(watches[watch].end, offset + fragment.backing.size());
          }
        }
        if (unavailable ||
            !ir::optimize(*translated, ir::fold_constants,
                          ir::simplify_identities, ir::fold_constants))
          break;
        identity->watch_pages = watches;
        identity->range_count = watch_count;
        for (std::size_t i = 0; i < watch_count; i++)
          identity->ranges[i] = {watches[i].base + watches[i].begin,
                                 watches[i].end - watches[i].begin};
        for (std::size_t i = 0; i < fragment_count; i++) {
          const let &fragment = fragments[i];
          const let index = identity->fragment_count++;
          identity->fragments[index] = fragment;
        }
        for (std::size_t i = 0; i < added_count; i++) {
          const let index = identity->page_count++;
          identity->pages[index] = added_pages[i];
          // A translated host page can span two RAM-relative version pages.
          for (std::size_t j = 0; j < fragment_count; j++)
            if ((fragments[j].pc & ~std::uint64_t{memory::page_size - 1}) ==
                added_pages[i]) {
              const let offset = fragments[j].pc & (memory::page_size - 1);
              identity->stamps[index] = memory_.stamp(
                  {fragments[j].backing.data() - offset, memory::page_size});
            }
          identity->versions[index] = identity->stamps[index].value();
        }
        for (std::size_t i = 0; i < count; i++) {
          const let bits = std::endian::native == std::endian::little
                               ? std::byteswap(words[i].bits)
                               : words[i].bits;
          std::memcpy(identity->source.data() + identity->bytes + i * 4, &bits,
                      sizeof(bits));
        }
        identity->bytes += count * 4;
        words_used += count;
        blocks.push_back(std::move(*translated));
        const let &block = blocks.back();
        if (const let branch_pc = block.branch_pc()) {
          const let bits = words[(*branch_pc - current_pc) / 4].bits;
          const let op = bits >> 26;
          const bool jalr =
              op == 0 && (bits & 63) == 9 && ((bits >> 11) & 31) == 31;
          if (op == 3 || jalr) {
            if (return_depth == returns.size())
              break;
            returns[return_depth++] = *branch_pc + 8;
          }
          if (op == 0 && (bits & 63) == 8 && return_depth) {
            current_pc = returns[--return_depth];
            continue;
          }
          if ((op == 2 || op == 3) && block.jump()) {
            current_pc = block.jump()->target;
            continue;
          }
        }
        const let *observed = find_cached(current_pc);
        if (!observed)
          break;
        if (!observed->successor_observed || observed->first_words != count ||
            observed->first_successor_pc % 4)
          break;
        current_pc = observed->first_successor_pc;
        if (block.indirect() && return_depth &&
            current_pc == returns[return_depth - 1])
          return_depth--;
      }
      if (blocks.size() < 2 || blocks.front().info().start_pc != pc ||
          !source_disjoint(
              std::span{identity->ranges}.first(identity->range_count), state))
        return false;
      let code = compiler_.compile_trace(blocks, arena, true, true, true,
                                          identity->backedge);
      if (!code)
        return std::unexpected(dispatch_error{
            dispatch_error_code::compilation, pc, {}, {}, code.error()});
      identity->blocks = blocks.size();
      identity->disjoint_state = &state;
      entry.code.emplace(std::move(*code));
      entry.instructions = blocks.front().info().instruction_count;
      entry.region = std::move(identity);
      entry.speculative_source = true;
      entry.cold = false;
      entry.fused = true;
      entry.tracing_attempted = true;
      entry.generation = memory_.mapping_generation();
      entry.validated_slice = 0;
      entry.hits = 0;
      entry.successors = {};
      region_compilations_++;
      largest_region_ = std::max(largest_region_, blocks.size());
      if (!valid_region(entry))
        return std::unexpected(
            dispatch_error{dispatch_error_code::code_changed, pc});
      return true;
    } catch (const std::bad_alloc &) {
      return std::unexpected(
          dispatch_error{dispatch_error_code::out_of_memory, pc});
    }
  }

  fn compile(cached_block &entry, const mips64::state &state,
               std::uint64_t pc, bool dump_ir, std::uint64_t limit,
               bool trace = false, bool allow_cold = true) noexcept
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
      if ((op == 0 && ((*fetched & 63) == 0x0c || (*fetched & 63) == 0x0d ||
                       (*fetched & 63) == 0x0f)) ||
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
        return compile_region(entry, state, pc);
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
    std::optional<count_batch::loop> count_pattern;
    if (memory_.may_speculate_translation() && instructions == 1 &&
        pc <= UINT64_MAX - 16 &&
        (words[0].bits & ~(31u << 16)) == 0x40004800u) {
      bool available = true;
      for (std::size_t i = 1; i < 5; i++) {
        const let backing = memory_.executable_tail({pc + i * 4});
        if (!backing || backing->size() < 4) {
          available = false;
          break;
        }
        std::uint32_t bits;
        std::memcpy(&bits, backing->data(), sizeof(bits));
        words[i] = {std::endian::native == std::endian::little
                        ? std::byteswap(bits)
                        : bits};
      }
      if (available)
        count_pattern = count_batch::decode(std::span{words}.first(5));
    }
    const let source_words = count_pattern ? 5u : instructions;
    std::array<const std::byte *, 2> host{first->data(), nullptr};
    if (source_words * 4 > first->size()) {
      const let second = memory_.executable_tail({pc + first->size()});
      if (!second)
        return std::unexpected(dispatch_error{
            dispatch_error_code::memory, pc + first->size(), second.error()});
      host[1] = second->data();
    }
    const let bytes = source_words * 4;
    const let first_bytes = std::min<std::size_t>(bytes, first->size());
    entry.region.reset();
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
    entry.source_words = source_words;
    entry.count_pattern = count_pattern;
    entry.speculative_source = trace || count_pattern.has_value();
    entry.cold = cold;
    entry.fused = trace;
    if (!trace) {
      entry.successor_observed = false;
      entry.first_words = 0;
    }
    for (std::size_t i = 0; i < source_words; i++) {
      const let bits = words[i].bits;
      entry.source[i].bits = std::endian::native == std::endian::little
                                 ? std::byteswap(bits)
                                 : bits;
    }
    entry.code.reset();
    if (!cold)
      entry.code.emplace(std::move(native->code));
    if (!revalidate(entry, pc))
      return std::unexpected(
          dispatch_error{dispatch_error_code::code_changed, pc});
    return !cold;
  }
};
} // namespace jit
