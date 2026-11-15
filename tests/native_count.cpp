#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <memory>
#include <memory_resource>
#include <print>
#include <span>
#include <string_view>
#include <utility>
#include <vector>
#include <dynamips/macros.h>
import jit.amd64;
import jit.ir;
import jit.memory;
import jit.mips64;

namespace {
namespace amd = jit::amd64;
namespace cpu = jit::mips64;
namespace mem = jit::memory;
fn require(bool valid, std::string_view text) -> void {
  if (!valid) {
    std::println(stderr, "Count entry: {}", text);
    std::exit(1);
  }
}

template <std::size_t N>
fn compile(const amd::compiler &compiler, std::pmr::memory_resource &arena,
             std::uint64_t pc, const std::array<std::uint32_t, N> &bits)
    -> amd::compiled_block {
  std::array<cpu::instruction_word, N> words;
  for (std::size_t i = 0; i < N; i++)
    words[i] = {bits[i]};
  let block = cpu::translate(words, {pc}, arena);
  require(block.has_value(), "translate test program");
  require(jit::ir::optimize(*block, jit::ir::fold_constants,
                            jit::ir::simplify_identities)
              .has_value(),
          "optimize test program");
  let native = compiler.compile(*block, arena, true);
  require(native.has_value(), "compile test program");
  return std::move(*native);
}
fn signed_word(std::uint64_t value) -> std::uint64_t {
  return static_cast<std::uint64_t>(std::int64_t{
      std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(value))});
}
} // namespace

fn main() -> int {
  let compiler = amd::compiler::create();
  require(compiler.has_value(), "create compiler");
  std::vector<std::byte> scratch(1024 * 1024);
  std::pmr::monotonic_buffer_resource arena{scratch.data(), scratch.size(),
                                            std::pmr::null_memory_resource()};
  std::array<std::byte, 2 * mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 3);
  constexpr let rwx =
      mem::permission::read | mem::permission::write | mem::permission::execute;
  require(memory && (*memory)->map_ram({0x1000}, {0}, rwx).has_value(),
          "map test RAM");
  const std::array code{std::span<const std::byte>{ram}.first(4)};
  cpu::state state{};
  amd::compiled_block::execution_session session{state, **memory};
  let original = compile(*compiler, arena, 0x8000, std::array{0x40024800u});
  let count = std::move(original);
  // The moved-from block explicitly promises an unavailable result.
  // NOLINTNEXTLINE(bugprone-use-after-move)
  const let moved_source = original.execute(session, code, 1, 1);
  require(!moved_source &&
              moved_source.error().code == amd::error_code::unavailable,
          "moved-from Count entries remain unavailable");
  let wide = compile(*compiler, arena, 0x8100, std::array{0x40224800u});
  let fault = compile(*compiler, arena, 0x8200, std::array{0x8c640000u});
  let trap = compile(*compiler, arena, 0x8300, std::array{0x20640001u});
  let store =
      compile(*compiler, arena, 0x8400, std::array{0xac620000u, 0x24040063u});
  let load =
      compile(*compiler, arena, 0x8500, std::array{0x8c640000u, 0x24050009u});
  unsigned callbacks = 0;
  mem::device endpoint{&callbacks,
                       [](void *opaque, mem::device_access) noexcept
                           -> std::expected<std::uint64_t, mem::device_error> {
                         (*static_cast<unsigned *>(opaque))++;
                         return 42;
                       },
                       nullptr};
  require(
      (*memory)
          ->map_device({0x5000}, {0x1000000}, endpoint, mem::permission::read)
          .has_value(),
      "map observed callback");
  for (const std::uint64_t value :
       {0ULL, 0x80000000ULL, 0x1234567880000001ULL}) {
    state.cop0.registers[9] = value;
    const let result = count.execute(session, code, 1, 1, 0);
    require(result && result->next_pc == 0x8004 && result->instructions == 1 &&
                result->blocks == 1 && result->accounted_instructions == 0 &&
                result->reason == amd::exit_reason::completed &&
                state.registers[2] == signed_word(value) &&
                state.cop0.registers[9] == value,
            "single MFC0 Count reads signed low32 without advancing time");
    const let full = wide.execute(session, code, 1, 1);
    require(full && full->next_pc == 0x8104 && full->instructions == 1 &&
                full->accounted_instructions == 0 &&
                state.registers[2] == value && state.cop0.registers[9] == value,
            "single DMFC0 Count preserves full64 value");
  }
  for (const bool fail : {false, true}) {
    state.registers[3] = fail ? 0xdead0000 : 0x7fffffff;
    const let dirty = fail ? fault.execute(session, code, 1, 1)
                           : trap.execute(session, code, 1, 1);
    require(
        fail ? (!dirty && dirty.error().code == amd::error_code::memory_fault)
             : (dirty && dirty->reason == amd::exit_reason::guest_exception),
        "leave a stale fault or trap in the reusable session");
    state.cop0.registers[9] = 0x87654321;
    const let counted = count.execute(session, code, 1, 1);
    require(counted && counted->reason == amd::exit_reason::completed &&
                counted->instructions == 1 &&
                counted->accounted_instructions == 0 &&
                state.registers[2] == signed_word(0x87654321),
            "Count execution ignores stale session fault/trap metadata");
    state.registers[3] = 0x1000;
    state.registers[5] = 0;
    const let restored = load.execute(session, code, 2, 1);
    require(restored && restored->reason == amd::exit_reason::completed &&
                restored->instructions == 2 && state.registers[5] == 9,
            "the following generic call resets stale fault/trap metadata");
  }
  state.registers[2] = 0x12345678;
  state.registers[3] = 0x1000;
  state.registers[4] = 0;
  const let modified = store.execute(session, code, 2, 1);
  require(modified && modified->instructions == 1 &&
              modified->next_pc == 0x8404 && state.registers[4] == 0,
          "leave a monitored code-write early exit in the session");
  state.cop0.registers[9] = 123;
  const let after_store = count.execute(session, code, 1, 1);
  require(after_store && after_store->instructions == 1 &&
              after_store->reason == amd::exit_reason::completed &&
              state.registers[2] == 123,
          "Count execution ignores stale monitored-write metadata");
  state.registers[5] = 0;
  const let after_generic = load.execute(session, code, 2, 1);
  require(after_generic && after_generic->instructions == 2 &&
              state.registers[4] == 0x12345678 && state.registers[5] == 9,
          "the following generic call resets monitored-write metadata");
  const let no_budget = count.execute(session, code, 0, 1);
  const let no_blocks = count.execute(session, code, 1, 0);
  const let no_code = count.execute(session, {}, 1, 1);
  const let no_cp0 = count.execute(state.registers, **memory, nullptr, nullptr,
                                   nullptr, code, 1, 1);
  const let short_state =
      count.execute(std::span{state.registers}.first(1), **memory, &state.cop0,
                    nullptr, nullptr, code, 1, 1);
  require(no_budget && no_budget->instructions == 0 && no_blocks &&
              no_blocks->instructions == 0 && !no_code &&
              no_code.error().code == amd::error_code::invalid_state &&
              !no_cp0 &&
              no_cp0.error().code == amd::error_code::cop0_required &&
              !short_state &&
              short_state.error().code == amd::error_code::invalid_state,
          "Count fast entry preserves code/state/CP0 and budget preconditions");
  let status = compile(*compiler, arena, 0x8580, std::array{0x40026000u});
  state.cop0.registers[12] = 0x87654321;
  const let other_cp0 = status.execute(session, code, 1, 1);
  require(other_cp0 && other_cp0->instructions == 1 &&
              state.registers[2] == signed_word(0x87654321),
          "other CP0 registers preserve their generic signed read semantics");
  let prefixed =
      compile(*compiler, arena, 0x8600, std::array{0x24040001u, 0x40024800u});
  count = std::move(prefixed);
  state.cop0.registers[9] = 13;
  state.cop0.half_cycle = true;
  const let prefix = count.execute(session, code, 2, 1);
  require(
      prefix && prefix->instructions == 2 &&
          prefix->accounted_instructions == 1 && state.registers[2] == 14 &&
          state.cop0.registers[9] == 14 && !state.cop0.half_cycle,
      "move assignment to a prefixed Count block restores generic accounting");
  require(callbacks == 0,
          "Count and RAM operations invoke no unrelated callback");
  std::println("Count entry: values, stale sessions, moves, limits and generic "
               "fallbacks passed");
}
