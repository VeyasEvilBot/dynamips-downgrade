#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory_resource>
#include <print>
#include <span>
#include <string_view>
#include <vector>

#include <dynamips/macros.h>

import jit.ir;
import jit.mips64;

namespace {
namespace ir = jit::ir;
namespace mips = jit::mips64;

fn run(std::span<const mips::instruction_word> words) -> int {
  constexpr mips::guest_address start{0x1000};
  alignas(std::max_align_t) std::array<std::byte, 64 * 1024> storage;
  std::pmr::monotonic_buffer_resource memory{storage.data(), storage.size(),
                                             std::pmr::null_memory_resource()};
  let translated = mips::translate(words, start, memory);
  if (!translated) {
    const let error = translated.error();
    const let message = [&]() -> std::string_view {
      switch (error.code) {
      case mips::error_code::unaligned_pc:
        return "unaligned PC";
      case mips::error_code::block_too_large:
        return "block too large";
      case mips::error_code::unsupported_instruction:
        return "unsupported instruction";
      case mips::error_code::invalid_encoding:
        return "invalid instruction encoding";
      case mips::error_code::out_of_memory:
        return "64 KiB IR scratch area exhausted; use a smaller instruction "
               "block";
      case mips::error_code::invalid_ir:
        return "IR construction failed";
      }
      return "translation failed";
    }();
    std::println(stderr, "jit: {} at instruction {} (PC {:#x})", message,
                 error.instruction_index,
                 start.value + error.instruction_index * 4);
    return 1;
  }
  translated->print("before optimization");
  let optimized = ir::optimize(*translated, ir::fold_constants);
  if (!optimized) {
    std::println(stderr, "jit: optimization failed at node {} (error {})",
                 optimized.error().node_index,
                 static_cast<unsigned>(optimized.error().code));
    return 1;
  }
  translated->print("after optimization");
  return 0;
}
} // namespace

fn main(int argc, char **argv) -> int {
  try {
    if (argc == 2 && std::string_view{argv[1]} == "--help") {
      std::println(
          "Usage: jit-playground [HEX_WORD ...]\n"
          "With no words, translate the built-in register-only example.\n"
          "Example: jit-playground 3401002a 3402000f 00221826 0062202d");
      return 0;
    }
    if (argc == 1) {
      constexpr std::array words{mips::instruction_word{0x3401002a},
                                 mips::instruction_word{0x3402000f},
                                 mips::instruction_word{0x00221826},
                                 mips::instruction_word{0x0062202d},
                                 mips::instruction_word{0x34a60001}};
      return run(words);
    }
    std::vector<mips::instruction_word> words;
    words.reserve(static_cast<std::size_t>(argc - 1));
    for (int i = 1; i < argc; i++) {
      std::string_view text{argv[i]};
      if (text.starts_with("0x") || text.starts_with("0X"))
        text.remove_prefix(2);
      std::uint32_t bits = 0;
      const let parsed =
          std::from_chars(text.data(), text.data() + text.size(), bits, 16);
      if (text.empty() || parsed.ec != std::errc{} ||
          parsed.ptr != text.data() + text.size()) {
        std::println(stderr, "jit: invalid 32-bit hexadecimal word: {}",
                     argv[i]);
        return 2;
      }
      words.push_back({bits});
    }
    return run(words);
  } catch (const std::exception &error) {
    std::println(stderr, "jit: {}", error.what());
    return 1;
  }
}
