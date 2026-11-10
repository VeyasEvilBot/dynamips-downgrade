#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory_resource>
#include <optional>
#include <print>
#include <span>
#include <string_view>
#include <utility>

#include <dynamips/macros.h>

import jit.ir;
import jit.mips64;
import jit.amd64;

namespace {
namespace ir = jit::ir;
namespace mips = jit::mips64;
namespace amd64 = jit::amd64;

fn require(bool condition, std::string_view message) -> void {
  if (!condition) {
    std::println(stderr, "jit test: {}", message);
    std::exit(1);
  }
}

fn next(std::uint64_t &seed) -> std::uint64_t {
  seed ^= seed << 13;
  seed ^= seed >> 7;
  seed ^= seed << 17;
  return seed;
}

fn differential() -> void {
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create compiler");
  std::uint64_t seed = 0x8e097c334bd12a65;
  for (unsigned trial = 0; trial < 128; trial++) {
    std::array<mips::instruction_word, 128> words;
    std::array<std::uint64_t, 32> initial;
    for (let &value : initial)
      value = next(seed);
    let expected = initial;
    const let read = [&](std::uint32_t index) {
      return index == 0 ? std::uint64_t{0} : expected[index];
    };
    for (let &word : words) {
      const let rs = static_cast<std::uint32_t>(next(seed) & 31);
      const let rt = static_cast<std::uint32_t>(next(seed) & 31);
      const let rd = static_cast<std::uint32_t>(next(seed) & 31);
      switch (next(seed) % 3) {
      case 0: {
        const let immediate = static_cast<std::uint32_t>(next(seed) & 0xffff);
        word.bits = (0x0du << 26) | (rs << 21) | (rt << 16) | immediate;
        if (rt != 0)
          expected[rt] = read(rs) | immediate;
        break;
      }
      case 1:
        word.bits = (rs << 21) | (rt << 16) | (rd << 11) | 0x26;
        if (rd != 0)
          expected[rd] = read(rs) ^ read(rt);
        break;
      case 2:
        word.bits = (rs << 21) | (rt << 16) | (rd << 11) | 0x2d;
        if (rd != 0)
          expected[rd] = read(rs) + read(rt);
        break;
      }
    }
    alignas(std::max_align_t) std::array<std::byte, 64 * 1024> storage;
    std::pmr::monotonic_buffer_resource memory{
        storage.data(), storage.size(), std::pmr::null_memory_resource()};
    let block = mips::translate(words, {0x1000}, memory);
    require(block.has_value(), "translate generated instructions");
    for (unsigned pass = 0; pass < 2; pass++) {
      if (pass == 1)
        require(ir::optimize(*block, ir::fold_constants).has_value(),
                "optimize generated IR");
      alignas(std::max_align_t) std::array<std::byte, 64 * 1024> scratch;
      std::pmr::monotonic_buffer_resource native_memory{
          scratch.data(), scratch.size(), std::pmr::null_memory_resource()};
      let compiled = compiler->compile(*block, native_memory);
      require(compiled.has_value(), "compile generated IR");
      let actual = initial;
      const let result = compiled->run(actual);
      require(result && *result == 0x1200, "native next PC");
      require(actual == expected, "native registers match MIPS oracle");
    }
  }
}

fn ownership_and_errors() -> void {
  std::optional<amd64::compiled_block> retained;
  {
    let compiler = amd64::compiler::create();
    require(compiler.has_value(), "create lifetime compiler");
    alignas(std::max_align_t) std::array<std::byte, 4096> storage;
    std::pmr::monotonic_buffer_resource memory{
        storage.data(), storage.size(), std::pmr::null_memory_resource()};
    constexpr std::array words{mips::instruction_word{0x0022182d},
                               mips::instruction_word{0x00222026}};
    let block = mips::translate(words, {0xfffffffffffffff4}, memory);
    require(block.has_value(), "translate wrapping add and reused operand");
    let exhausted =
        compiler->compile(*block, *std::pmr::null_memory_resource());
    require(!exhausted &&
                exhausted.error().code == amd64::error_code::out_of_memory,
            "bounded compilation scratch exhaustion");
    let native = compiler->compile(*block, memory);
    require(native.has_value(), "compile lifetime block");
    retained.emplace(std::move(*native));
    std::array<std::uint64_t, 32> state{};
    require(!native->run(state), "moved-from block rejects execution");
    let replacement = compiler->compile(*block, memory);
    require(replacement.has_value(), "compile move-assignment destination");
    *replacement = std::move(*retained);
    *replacement = std::move(*replacement);
    retained.emplace(std::move(*replacement));

    block->edit_nodes()[0].slot.index = 32;
    let invalid = compiler->compile(*block, memory);
    require(!invalid && invalid.error().code == amd64::error_code::invalid_ir,
            "reject invalid IR before emitting");
  }
  std::array<std::uint64_t, 32> state{};
  state[1] = UINT64_MAX;
  state[2] = 1;
  let short_state = retained->run(std::span{state}.first(31));
  require(!short_state &&
              short_state.error().code == amd64::error_code::invalid_state,
          "reject short state before executing");
  require(state[3] == 0 && state[4] == 0,
          "rejected call leaves state unchanged");
  let result = retained->run(state);
  require(result && *result == 0xfffffffffffffffc && state[3] == 0 &&
              state[4] == (UINT64_MAX ^ 1u),
          "native code survives compiler, IR and scratch destruction");
  state[1] = 3;
  require(retained->run(state).has_value() && state[3] == 4 && state[4] == 2,
          "reuse native code with changed state");
}
} // namespace

fn main() -> int {
  differential();
  ownership_and_errors();
  std::println("jit: native differential and ownership checks passed");
}
