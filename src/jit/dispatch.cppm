module;

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <exception>
#include <memory>
#include <memory_resource>
#include <new>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <dynamips/macros.h>

export module jit.dispatch;

import jit.ir;
import jit.mips64;
import jit.amd64;
import jit.memory;

export namespace jit {
enum class dispatch_error_code {
  invalid_image,
  invalid_pc,
  code_changed,
  writable_code,
  translation,
  compilation,
  memory,
  out_of_memory,
  diagnostic
};

struct dispatch_error {
  dispatch_error_code code;
  std::uint64_t pc;
  memory::fault access{};
  mips64::error translation{};
  amd64::error compilation{};
  std::optional<std::uint64_t> branch_pc{};
};
enum class stop_reason { completed, budget_exhausted, yielded };

struct dispatch_result {
  stop_reason reason;
  std::uint64_t instructions;
  std::size_t blocks;
  std::size_t compiled;
};

// One owner, immutable instruction image. RAM and the address space outlive
// this object; direct host writes to executable backing are forbidden while in
// use.
class dispatcher {
public:
  dispatcher(const dispatcher &) = delete;
  fn operator=(const dispatcher &) -> dispatcher & = delete;

  [[nodiscard]] static fn
  create(std::span<const mips64::instruction_word> image,
         mips64::guest_address start, memory::address_space &memory) noexcept
      -> std::expected<std::unique_ptr<dispatcher>, dispatch_error> {
    if (start.value % 4 || image.size() > 65536 ||
        image.size() > (UINT64_MAX - start.value) / 4)
      return std::unexpected(
          dispatch_error{dispatch_error_code::invalid_image, start.value});
    let compiler = amd64::compiler::create();
    if (!compiler)
      return std::unexpected(dispatch_error{dispatch_error_code::compilation,
                                            start.value,
                                            {},
                                            {},
                                            compiler.error()});
    try {
      let result = std::unique_ptr<dispatcher>{
          new dispatcher{start, memory, std::move(*compiler)}};
      result->image_.assign(image.begin(), image.end());
      result->cache_.resize(image.size());
      if (!memory.seal_executable_pages())
        return std::unexpected(
            dispatch_error{dispatch_error_code::writable_code, start.value});
      return result;
    } catch (const std::bad_alloc &) {
      return std::unexpected(
          dispatch_error{dispatch_error_code::out_of_memory, start.value});
    }
  }

  [[nodiscard]] fn run_slice(mips64::state &state, std::uint64_t budget,
                              std::size_t block_limit = 64,
                              bool dump_ir = false) noexcept
      -> std::expected<dispatch_result, dispatch_error> {
    dispatch_result result{stop_reason::yielded, 0, 0, 0};
    const let end = start_.value + image_.size() * 4;
    while (true) {
      if (state.pc.value == end) {
        result.reason = stop_reason::completed;
        return result;
      }
      if (state.pc.value < start_.value || state.pc.value >= end ||
          state.pc.value % 4)
        return std::unexpected(
            dispatch_error{dispatch_error_code::invalid_pc, state.pc.value});
      if (result.blocks == block_limit)
        return result;
      if (result.instructions == budget) {
        result.reason = stop_reason::budget_exhausted;
        return result;
      }
      if (generation_ != memory_.mapping_generation()) {
        for (let &entry : cache_)
          entry.code.reset();
        generation_ = memory_.mapping_generation();
      }
      const let index =
          static_cast<std::size_t>((state.pc.value - start_.value) / 4);
      let &entry = cache_[index];
      if (!entry.code) {
        const let compiled = compile(index, dump_ir);
        if (!compiled) {
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
          entry.code->run(state.registers, memory_, &state.cop0);
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
  }

private:
  struct cached_block {
    std::optional<amd64::compiled_block> code;
    std::uint32_t instructions = 0;
  };

  mips64::guest_address start_;
  memory::address_space &memory_;
  amd64::compiler compiler_;
  std::vector<mips64::instruction_word> image_;
  std::vector<cached_block> cache_;
  std::uint64_t generation_;
  alignas(std::max_align_t) std::array<std::byte, 64 * 1024> scratch_;

  dispatcher(mips64::guest_address start, memory::address_space &memory,
             amd64::compiler compiler) noexcept
      : start_(start), memory_(memory), compiler_(std::move(compiler)),
        generation_(memory.mapping_generation()) {}

  fn compile(std::size_t index, bool dump_ir) noexcept
      -> std::expected<void, dispatch_error> {
    let count = std::min<std::size_t>(64, image_.size() - index);
    for (std::size_t i = 0; i < count; i++) {
      if (!mips64::has_delay_slot(image_[index + i]))
        continue;
      count = std::min(i + 2, image_.size() - index);
      break;
    }
    const let pc = start_.value + index * 4;
    for (std::size_t i = 0; i < count; i++) {
      const let fetched = memory_.fetch<std::uint32_t>({pc + i * 4});
      if (!fetched) {
        const bool delay_slot =
            i != 0 && mips64::has_delay_slot(image_[index + i - 1]);
        const let prefix = delay_slot ? i - 1 : i;
        if (prefix != 0) {
          count = prefix;
          break;
        }
        return std::unexpected(dispatch_error{
            dispatch_error_code::memory,
            pc + i * 4,
            fetched.error(),
            {},
            {},
            delay_slot ? std::optional<std::uint64_t>{pc + (i - 1) * 4}
                       : std::nullopt});
      }
      if (*fetched != image_[index + i].bits)
        return std::unexpected(
            dispatch_error{dispatch_error_code::code_changed, pc + i * 4});
    }
    std::pmr::monotonic_buffer_resource arena{scratch_.data(), scratch_.size(),
                                              std::pmr::null_memory_resource()};
    let translated =
        mips64::translate(std::span{image_}.subspan(index, count), {pc}, arena);
    if (!translated)
      return std::unexpected(
          dispatch_error{dispatch_error_code::translation,
                         pc + translated.error().instruction_index * 4,
                         {},
                         translated.error()});
    if (let optimized = ir::optimize(*translated, ir::fold_constants);
        !optimized)
      return std::unexpected(
          dispatch_error{dispatch_error_code::translation, pc});
    // Printing is diagnostic work, never part of cached execution.
    if (dump_ir) {
      try {
        translated->print("compiled block");
      } catch (const std::bad_alloc &) {
        return std::unexpected(
            dispatch_error{dispatch_error_code::out_of_memory, pc});
      } catch (const std::exception &) {
        return std::unexpected(
            dispatch_error{dispatch_error_code::diagnostic, pc});
      }
    }
    let compiled = compiler_.compile(*translated, arena);
    if (!compiled)
      return std::unexpected(dispatch_error{
          dispatch_error_code::compilation, pc, {}, {}, compiled.error()});
    cache_[index].instructions = translated->info().instruction_count;
    cache_[index].code.emplace(std::move(*compiled));
    return {};
  }
};
} // namespace jit
