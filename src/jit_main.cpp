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
import jit.amd64;
import jit.memory;
import jit.uart;

namespace {
namespace ir = jit::ir;
namespace mips = jit::mips64;

fn fault_name(jit::memory::fault_reason reason) noexcept -> std::string_view {
  switch (reason) {
  case jit::memory::fault_reason::misaligned:
    return "misaligned access";
  case jit::memory::fault_reason::unmapped:
    return "unmapped address";
  case jit::memory::fault_reason::permission:
    return "permission denied";
  case jit::memory::fault_reason::unsupported_size:
    return "unsupported access width";
  case jit::memory::fault_reason::bus:
    return "device bus fault";
  }
  return "unknown memory fault";
}

fn run(std::span<const mips::instruction_word> words, bool dump_ir,
         bool registers) -> int {
  constexpr mips::guest_address start{0x1000};
  if (words.size() > jit::memory::page_size / sizeof(std::uint32_t)) {
    std::println(stderr,
                 "jit: program exceeds the 1024-instruction playground limit");
    return 2;
  }
  std::array<std::byte, 2 * jit::memory::page_size> ram{};
  jit::uart console{{0x100000}};
  let address_space = jit::memory::address_space::create(ram, 3);
  if (!address_space) {
    std::println(stderr, "jit: cannot reserve guest memory");
    return 1;
  }
  let &guest = **address_space;
  constexpr let rw =
      jit::memory::permission::read | jit::memory::permission::write;
  if (!guest.map_ram({start.value}, {jit::memory::page_size}, rw) ||
      !guest.map_ram({0x2000}, {0}, rw) ||
      !guest.map_device({0x8000}, {0x100000}, console.endpoint(), rw)) {
    std::println(stderr, "jit: cannot map playground RAM and MMIO");
    return 1;
  }
  for (std::size_t i = 0; i < words.size(); i++)
    if (!guest.store<std::uint32_t>({start.value + i * 4}, words[i].bits)) {
      std::println(stderr, "jit: cannot load program into guest RAM");
      return 1;
    }
  if (!guest.protect({start.value}, jit::memory::permission::read |
                                        jit::memory::permission::execute)) {
    std::println(stderr, "jit: cannot protect guest code");
    return 1;
  }
  std::array<mips::instruction_word,
             jit::memory::page_size / sizeof(std::uint32_t)>
      fetched;
  for (std::size_t i = 0; i < words.size(); i++) {
    let instruction = guest.fetch<std::uint32_t>({start.value + i * 4});
    if (!instruction) {
      std::println(stderr, "jit: cannot fetch guest instruction at {:#x}",
                   start.value + i * 4);
      return 1;
    }
    fetched[i].bits = *instruction;
  }
  alignas(std::max_align_t) std::array<std::byte, 64 * 1024> storage;
  std::pmr::monotonic_buffer_resource memory{storage.data(), storage.size(),
                                             std::pmr::null_memory_resource()};
  let translated =
      mips::translate(std::span{fetched}.first(words.size()), start, memory);
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
  if (dump_ir)
    translated->print("before optimization");
  let optimized = ir::optimize(*translated, ir::fold_constants);
  if (!optimized) {
    std::println(stderr, "jit: optimization failed at node {} (error {})",
                 optimized.error().node_index,
                 static_cast<unsigned>(optimized.error().code));
    return 1;
  }
  if (dump_ir)
    translated->print("after optimization");
  let compiler = jit::amd64::compiler::create();
  if (!compiler) {
    std::println(stderr, "jit: {}", jit::amd64::message(compiler.error()));
    return 1;
  }
  alignas(std::max_align_t) std::array<std::byte, 64 * 1024> native_storage;
  std::pmr::monotonic_buffer_resource scratch{native_storage.data(),
                                              native_storage.size(),
                                              std::pmr::null_memory_resource()};
  let native = compiler->compile(*translated, scratch);
  if (!native) {
    const let error = native.error();
    std::println(stderr, "jit: {} at node {}", jit::amd64::message(error),
                 error.node_index);
    return 1;
  }
  mips::state state{};
  state.pc = start;
  let executed = native->run(state.registers, guest);
  if (!executed) {
    const let &error = executed.error();
    std::println(stderr, "jit: {} at PC {:#x}, address {:#x}",
                 fault_name(error.access.reason), error.pc,
                 error.access.address.value);
    state.pc.value = error.pc;
    if (!console.output().empty())
      std::println("MMIO console before fault:\n{}", console.output());
    if (registers)
      state.print();
    return 1;
  }
  state.pc.value = *executed;
  std::println("Executed {} instructions; PC={:#x}", words.size(),
               state.pc.value);
  std::println("MMIO console:\n{}",
               console.output().empty() ? "(no output)" : console.output());
  if (registers)
    state.print();
  return 0;
}
} // namespace

fn main(int argc, char **argv) -> int {
  try {
    bool dump_ir = false;
    bool registers = false;
    std::vector<mips::instruction_word> words;
    words.reserve(static_cast<std::size_t>(argc - 1));
    for (int i = 1; i < argc; i++) {
      std::string_view text{argv[i]};
      if (text == "--help") {
        std::println(
            "Usage: jit-playground [--dump-ir] [--registers] [HEX_WORD ...]\n"
            "Run the MMIO hello example, or a straight-line MIPS program.\n"
            "Code: 0x1000; RAM: 0x2000; MMIO TX: 0x8000; status: 0x8004.\n"
            "Supported: ORI, XOR, DADDU, LW, SW, LD, SD. Registers start at "
            "zero.");
        return 0;
      }
      if (text == "--dump-ir") {
        dump_ir = true;
        continue;
      }
      if (text == "--registers") {
        registers = true;
        continue;
      }
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
    if (!words.empty())
      return run(words, dump_ir, registers);
    constexpr std::array example{
        mips::instruction_word{0x34018000}, mips::instruction_word{0x34020068},
        mips::instruction_word{0xac220000}, mips::instruction_word{0x34020065},
        mips::instruction_word{0xac220000}, mips::instruction_word{0x3402006c},
        mips::instruction_word{0xac220000}, mips::instruction_word{0x3402006c},
        mips::instruction_word{0xac220000}, mips::instruction_word{0x3402006f},
        mips::instruction_word{0xac220000}, mips::instruction_word{0x3402000a},
        mips::instruction_word{0xac220000}, mips::instruction_word{0x8c230004},
        mips::instruction_word{0x34042000}, mips::instruction_word{0xac820000},
        mips::instruction_word{0x8c850000}};
    return run(example, dump_ir, registers);
  } catch (const std::exception &error) {
    std::println(stderr, "jit: {}", error.what());
    return 1;
  }
}
