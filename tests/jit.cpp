#include <array>
#include <bit>
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
import jit.memory;
import jit.uart;
import jit.example;

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

fn immediate_operands() -> void {
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create immediate compiler");
  constexpr std::array<std::uint64_t, 8> constants{
      0,           0x7fffffff,         0x80000000,         0xffffffff,
      0x100000000, 0xffffffff7fffffff, 0xffffffff80000000, UINT64_MAX};
  constexpr std::array operations{ir::binary_kind::add_wrap,
                                  ir::binary_kind::bit_or,
                                  ir::binary_kind::bit_xor};
  for (const let bits : constants) {
    for (const let operation : operations) {
      for (unsigned side = 0; side < 2; side++) {
        alignas(std::max_align_t) std::array<std::byte, 64 * 1024> storage;
        std::pmr::monotonic_buffer_resource memory{
            storage.data(), storage.size(), std::pmr::null_memory_resource()};
        let block = ir::block::create({0x3000, 1, 4}, 5, memory);
        require(block.has_value(), "create immediate IR");
        const let append = [&](ir::node node) {
          let result = block->append(node);
          require(result.has_value(), "append immediate IR");
          return *result;
        };
        const let input = append(ir::node::read({1}, 0));
        const let constant = append(ir::node::constant(bits, 0));
        const let result =
            append(ir::node::binary(operation, side == 0 ? input : constant,
                                    side == 0 ? constant : input, 0));
        append(ir::node::write({2}, result, 0));
        append(ir::node::write({3}, constant, 0));
        let native = compiler->compile(*block, memory);
        require(native.has_value(), "compile immediate boundary");
        for (const let value : {std::uint64_t{0x123456789abcdef}, UINT64_MAX}) {
          std::array<std::uint64_t, 4> state{0, value, 0, 0};
          const let expected =
              operation == ir::binary_kind::add_wrap ? value + bits
              : operation == ir::binary_kind::bit_or ? value | bits
                                                     : value ^ bits;
          require(native->run(state).has_value() && state[2] == expected &&
                      state[3] == bits,
                  "immediate encoding and shared constant retain all bits");
        }
      }
    }
  }
}

fn observed_stores() -> void {
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create store compiler");
  for (unsigned observe = 0; observe < 3; observe++) {
    alignas(std::max_align_t) std::array<std::byte, 64 * 1024> storage;
    std::pmr::monotonic_buffer_resource memory{
        storage.data(), storage.size(), std::pmr::null_memory_resource()};
    let block = ir::block::create({0x4000, 1, 4}, 8, memory);
    require(block.has_value(), "create store IR");
    const let append = [&](ir::node node) {
      let result = block->append(node);
      require(result.has_value(), "append store IR");
      return *result;
    };
    const let first = append(ir::node::constant(11, 0));
    const let last = append(ir::node::constant(22, 0));
    append(ir::node::write({1}, first, 0));
    ir::value_id snapshot{};
    if (observe != 0)
      snapshot =
          append(ir::node::read({static_cast<std::uint16_t>(observe)}, 0));
    append(ir::node::write({1}, last, 0));
    if (observe != 0)
      append(ir::node::write({3}, snapshot, 0));
    let native = compiler->compile(*block, memory);
    require(native.has_value(), "compile observed stores");
    std::array<std::uint64_t, 4> state{0, 5, 33, 0};
    require(native->run(state).has_value() && state[1] == 22 &&
                state[2] == 33 &&
                state[3] == (observe == 0   ? 0u
                             : observe == 1 ? 11u
                                            : 33u),
            "eliminate overwritten stores without changing observed state");
  }
}

fn memory_operations() -> void {
  namespace mem = jit::memory;
  std::array<std::byte, mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 1);
  require(memory.has_value() &&
              (*memory)
                  ->map_ram({0x1000}, {0},
                            mem::permission::read | mem::permission::write)
                  .has_value(),
          "map JIT RAM");
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create memory compiler");
  constexpr std::array words{
      mips::instruction_word{0xac220000}, mips::instruction_word{0x8c230000},
      mips::instruction_word{0xfc240008}, mips::instruction_word{0xdc250008},
      mips::instruction_word{0x8ce6fffc}, mips::instruction_word{0xac200010},
      mips::instruction_word{0x8c200000}};
  for (unsigned pass = 0; pass < 2; pass++) {
    alignas(std::max_align_t) std::array<std::byte, 64 * 1024> storage;
    std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                              std::pmr::null_memory_resource()};
    let block = mips::translate(words, {0x1000}, arena);
    require(block.has_value(), "translate memory instructions");
    if (pass != 0)
      require(ir::optimize(*block, ir::fold_constants).has_value(),
              "optimize memory IR");
    let native = compiler->compile(*block, arena);
    require(native.has_value(), "compile memory IR");
    std::array<std::uint64_t, 32> state{};
    state[1] = 0x1000;
    state[2] = 0x1122334480000001;
    state[4] = 0xfedcba9876543210;
    state[7] = 0x1004;
    const let missing = native->run(state);
    require(!missing &&
                missing.error().code == amd64::error_code::memory_required,
            "reject memory block without address space");
    const let result = native->run(state, **memory);
    require(result && *result == 0x101c && state[0] == 0 &&
                state[3] == 0xffffffff80000001 && state[6] == state[3] &&
                state[5] == state[4],
            "native load/store widths, sign extension and signed offset");
    require(ram[0] == std::byte{0x80} && ram[3] == std::byte{1} &&
                (*memory)->load<std::uint64_t>({0x1010}) == 0,
            "guest stores use guest byte order and architectural zero");
  }
  constexpr std::array faulting{mips::instruction_word{0x34011111},
                                mips::instruction_word{0x8c620000},
                                mips::instruction_word{0x34012222}};
  for (unsigned discard = 0; discard < 2; discard++) {
    alignas(std::max_align_t) std::array<std::byte, 8192> storage;
    std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                              std::pmr::null_memory_resource()};
    let input = faulting;
    if (discard != 0)
      input[1].bits = 0x8c600000;
    let block = mips::translate(input, {0x1000}, arena);
    require(block.has_value(), "translate faulting load");
    require(ir::optimize(*block, ir::fold_constants).has_value(),
            "optimize faulting load");
    let native = compiler->compile(*block, arena);
    require(native.has_value(), "compile faulting load");
    std::array<std::uint64_t, 32> state{};
    state[2] = 0xaaaa;
    state[3] = 0x3000;
    const let result = native->run(state, **memory);
    require(!result && result.error().pc == 0x1004 &&
                result.error().access.address.value == 0x3000 &&
                result.error().access.reason == mem::fault_reason::unmapped &&
                state[1] == 0x1111 && state[2] == 0xaaaa && state[0] == 0,
            "fault exits preserve earlier writes, including loads to zero");
  }
  require((*memory)->protect({0x1000}, mem::permission::read).has_value(),
          "protect JIT RAM");
  alignas(std::max_align_t) std::array<std::byte, 8192> storage;
  std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                            std::pmr::null_memory_resource()};
  constexpr std::array store{mips::instruction_word{0xac220000},
                             mips::instruction_word{0x3403002a}};
  let block = mips::translate(store, {0x1000}, arena);
  require(block.has_value(), "translate faulting store");
  let native = compiler->compile(*block, arena);
  require(native.has_value(), "compile faulting store");
  std::array<std::uint64_t, 32> state{};
  state[1] = 0x1000;
  const let before = ram;
  const let result = native->run(state, **memory);
  require(!result && result.error().pc == 0x1000 &&
              result.error().access.reason == mem::fault_reason::permission &&
              result.error().access.access == mem::access_kind::write &&
              state[3] == 0 && ram == before,
          "faulting stores do not modify memory or later registers");
}

fn mmio_console() -> void {
  namespace mem = jit::memory;
  std::array<std::byte, mem::page_size> ram{};
  jit::uart console{{0x100000}};
  let memory = mem::address_space::create(ram, 2);
  require(memory.has_value(), "create MMIO address space");
  let &guest = **memory;
  constexpr let rw = mem::permission::read | mem::permission::write;
  require(guest.map_ram({0x2000}, {0}, rw).has_value() &&
              guest.map_device({0x8000}, {0x100000}, console.endpoint(), rw)
                  .has_value(),
          "map console and RAM");
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create MMIO compiler");
  alignas(std::max_align_t) std::array<std::byte, 64 * 1024> storage;
  std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                            std::pmr::null_memory_resource()};
  constexpr std::array words{
      mips::instruction_word{0x34018000}, mips::instruction_word{0x34020068},
      mips::instruction_word{0xac220000}, mips::instruction_word{0x34020065},
      mips::instruction_word{0xac220000}, mips::instruction_word{0x3402006c},
      mips::instruction_word{0xac220000}, mips::instruction_word{0x3402006c},
      mips::instruction_word{0xac220000}, mips::instruction_word{0x3402006f},
      mips::instruction_word{0xac220000}, mips::instruction_word{0x3402000a},
      mips::instruction_word{0xac220000}, mips::instruction_word{0x8c230004},
      mips::instruction_word{0x34042000}, mips::instruction_word{0xac820000},
      mips::instruction_word{0x8c850000}};
  let block = mips::translate(words, {0x1000}, arena);
  require(block.has_value() &&
              ir::optimize(*block, ir::fold_constants).has_value(),
          "translate and optimize MMIO program");
  let native = compiler->compile(*block, arena);
  require(native.has_value(), "compile MMIO program");
  std::array<std::uint64_t, 32> state{};
  const let result = native->run(state, guest);
  require(
      result && *result == 0x1044 && console.output() == "hello\n" &&
          state[3] == 1 && state[5] == 10 &&
          guest.load<std::uint32_t>({0x2000}) == 10,
      "native guest writes exact console bytes, reads status and accesses RAM");
  require(native->run(state, guest).has_value() &&
              console.output() == "hello\nhello\n",
          "repeated execution preserves every MMIO transaction");
  constexpr std::array faulting{
      mips::instruction_word{0x34018000}, mips::instruction_word{0x34020058},
      mips::instruction_word{0xac220000}, mips::instruction_word{0x8c200000},
      mips::instruction_word{0x34020059}, mips::instruction_word{0xac220000}};
  let fault_block = mips::translate(faulting, {0x1000}, arena);
  require(fault_block.has_value() &&
              ir::optimize(*fault_block, ir::fold_constants).has_value(),
          "translate faulting MMIO program");
  let fault_native = compiler->compile(*fault_block, arena);
  require(fault_native.has_value(), "compile faulting MMIO program");
  const let failed = fault_native->run(state, guest);
  require(
      !failed && failed.error().pc == 0x100c &&
          failed.error().access.reason == mem::fault_reason::bus &&
          state[2] == 'X' && console.output() == "hello\nhello\nX",
      "MMIO fault preserves earlier effects and suppresses later instructions");
  require(guest.store<std::uint32_t, std::endian::little>({0x8000}, 0x21000000)
                  .has_value() &&
              console.output() == "hello\nhello\nX!" &&
              guest.load<std::uint32_t, std::endian::little>({0x8004}) ==
                  0x01000000,
          "device register byte order is explicit");
  const let unsupported = guest.store<std::uint16_t>({0x8000}, 1);
  require(!unsupported &&
              unsupported.error().reason == mem::fault_reason::unsupported_size,
          "UART rejects unsupported access width");
  for (std::size_t i = console.output().size(); i < jit::uart::capacity; i++)
    require(guest.store<std::uint32_t>({0x8000}, 'A').has_value(),
            "fill bounded console");
  require(guest.load<std::uint32_t>({0x8004}) == 0,
          "full UART clears ready status");
  const let full = fault_native->run(state, guest);
  require(!full && full.error().pc == 0x1008 &&
              full.error().access.access == mem::access_kind::write &&
              console.output().size() == jit::uart::capacity,
          "full console faults without overwriting the buffer");
}
fn example_program() -> void {
  namespace mem = jit::memory;
  for (unsigned pass = 0; pass < 2; pass++) {
    std::array<std::byte, mem::page_size> ram{};
    jit::uart console{{0x100000}};
    let memory = mem::address_space::create(ram, 2);
    require(memory.has_value(), "create example memory");
    constexpr let rw = mem::permission::read | mem::permission::write;
    require((*memory)->map_ram({0x2000}, {0}, rw).has_value() &&
                (*memory)
                    ->map_device({0x8000}, {0x100000}, console.endpoint(), rw)
                    .has_value(),
            "map example RAM and console");
    alignas(std::max_align_t) std::array<std::byte, 64 * 1024> storage;
    std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                              std::pmr::null_memory_resource()};
    let block = mips::translate(jit::example::program, {0x1000}, arena);
    require(block.has_value(), "translate complex example");
    if (pass != 0)
      require(ir::optimize(*block, ir::fold_constants).has_value(),
              "optimize complex example");
    let compiler = amd64::compiler::create();
    require(compiler.has_value(), "create example compiler");
    let native = compiler->compile(*block, arena);
    require(native.has_value(), "compile complex example");
    std::array<std::uint64_t, 32> state{};
    const let result = native->run(state, **memory);
    require(result && *result == 0x1000 + jit::example::program.size() * 4 &&
                console.output() == "sum=6\n" && state[7] == 6 &&
                state[8] == 6 && state[10] == 0,
            "complex example computes and transmits the result");
    require((*memory)->load<std::uint64_t>({0x2000}) == 4 &&
                (*memory)->load<std::uint64_t>({0x2008}) == 2 &&
                (*memory)->load<std::uint64_t>({0x2010}) == 6 &&
                (*memory)->load<std::uint32_t>({0x2018}) == 0 &&
                (*memory)->load<std::uint32_t>({0x201c}) == 1 &&
                (*memory)->load<std::uint32_t>({0x2050}) == '6',
            "example leaves operands, sum, check, status and output in RAM");
  }
}

} // namespace

fn main() -> int {
  differential();
  ownership_and_errors();
  immediate_operands();
  observed_stores();
  memory_operations();
  mmio_console();
  example_program();
  std::println(
      "jit: native lowering, differential and ownership checks passed");
}
