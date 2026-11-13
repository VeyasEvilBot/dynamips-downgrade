#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <expected>
#include <format>
#include <string>
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
import jit.tlb;
import jit.uart;
import jit.example;
import jit.dispatch;
import jit.memory_dispatch;

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
      switch (next(seed) % 10) {
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
      case 9:
        word.bits = (rs << 21) | (rt << 16) | (rd << 11) | 0x24;
        if (rd != 0)
          expected[rd] = read(rs) & read(rt);
        break;
      case 8: {
        const let immediate = static_cast<std::uint32_t>(next(seed) & 0xffff);
        word.bits = (0x19u << 26) | (rs << 21) | (rt << 16) | immediate;
        if (rt != 0)
          expected[rt] = read(rs) + static_cast<std::uint64_t>(std::int64_t{
                                        static_cast<std::int16_t>(immediate)});
        break;
      }
      case 7: {
        const let immediate = static_cast<std::uint32_t>(next(seed) & 0xffff);
        word.bits = (0x0bu << 26) | (rs << 21) | (rt << 16) | immediate;
        if (rt != 0)
          expected[rt] = read(rs) < static_cast<std::uint64_t>(std::int64_t{
                                        static_cast<std::int16_t>(immediate)});
        break;
      }
      case 6: {
        const let immediate = static_cast<std::uint32_t>(next(seed) & 0xffff);
        word.bits = (0x0au << 26) | (rs << 21) | (rt << 16) | immediate;
        if (rt != 0)
          expected[rt] = std::bit_cast<std::int64_t>(read(rs)) <
                         std::int64_t{static_cast<std::int16_t>(immediate)};
        break;
      }
      case 5: {
        const let immediate = static_cast<std::uint32_t>(next(seed) & 0xffff);
        word.bits = (0x0cu << 26) | (rs << 21) | (rt << 16) | immediate;
        if (rt != 0)
          expected[rt] = read(rs) & immediate;
        break;
      }
      case 4: {
        const let immediate = static_cast<std::uint32_t>(next(seed) & 0xffff);
        word.bits = (0x09u << 26) | (rs << 21) | (rt << 16) | immediate;
        if (rt != 0) {
          const let sum = static_cast<std::uint32_t>(read(rs)) +
                          static_cast<std::uint32_t>(std::int32_t{
                              static_cast<std::int16_t>(immediate)});
          expected[rt] = static_cast<std::uint64_t>(
              std::int64_t{std::bit_cast<std::int32_t>(sum)});
        }
        break;
      }
      case 3: {
        const let immediate = static_cast<std::uint32_t>(next(seed) & 0xffff);
        word.bits = (0x0fu << 26) | (rt << 16) | immediate;
        if (rt != 0)
          expected[rt] = static_cast<std::uint64_t>(
              std::int64_t{std::bit_cast<std::int32_t>(immediate << 16)});
        break;
      }
      }
    }
    alignas(std::max_align_t) std::array<std::byte, 64 * 1024> storage;
    std::pmr::monotonic_buffer_resource memory{
        storage.data(), storage.size(), std::pmr::null_memory_resource()};
    let block = mips::translate(words, {0x1000}, memory);
    require(block.has_value(), "translate generated instructions");
    for (unsigned pass = 0; pass < 2; pass++) {
      if (pass == 1)
        require(
            ir::optimize(*block, ir::fold_constants, ir::simplify_identities)
                .has_value(),
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
  constexpr std::array operations{
      ir::binary_kind::unsigned_less, ir::binary_kind::signed_less,
      ir::binary_kind::bit_and,       ir::binary_kind::add_wrap,
      ir::binary_kind::bit_or,        ir::binary_kind::bit_xor};
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
              operation == ir::binary_kind::unsigned_less
                  ? static_cast<std::uint64_t>((side == 0 ? value : bits) <
                                               (side == 0 ? bits : value))
              : operation == ir::binary_kind::signed_less
                  ? static_cast<std::uint64_t>(
                        std::bit_cast<std::int64_t>(side == 0 ? value : bits) <
                        std::bit_cast<std::int64_t>(side == 0 ? bits : value))
              : operation == ir::binary_kind::bit_and  ? value & bits
              : operation == ir::binary_kind::add_wrap ? value + bits
              : operation == ir::binary_kind::bit_or   ? value | bits
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
              ir::optimize(*block, ir::fold_constants, ir::simplify_identities)
                  .has_value(),
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
              ir::optimize(*fault_block, ir::fold_constants,
                           ir::simplify_identities)
                  .has_value(),
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
fn word_shifts() -> void {
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create word shift compiler");
  for (const unsigned function : {0u, 2u, 3u}) {
    for (std::uint32_t amount = 0; amount < 32; amount++) {
      alignas(std::max_align_t) std::array<std::byte, 8192> storage;
      std::pmr::monotonic_buffer_resource arena{
          storage.data(), storage.size(), std::pmr::null_memory_resource()};
      const std::array words{mips::instruction_word{(1u << 16) | (2u << 11) |
                                                    (amount << 6) | function}};
      let block = mips::translate(words, {0x1000}, arena);
      require(block.has_value(), "translate word shift");
      let native = compiler->compile(*block, arena);
      require(native.has_value(), "compile word shift");
      for (const std::uint64_t bits :
           {0ULL, 0x80000000ULL, 0xffffffffULL, 0x1234567887654321ULL}) {
        std::array<std::uint64_t, 32> state{};
        state[1] = bits;
        const let low = static_cast<std::uint32_t>(bits);
        const let word = function == 0 ? low << amount
                         : function == 2
                             ? low >> amount
                             : std::bit_cast<std::uint32_t>(
                                   std::bit_cast<std::int32_t>(low) >> amount);
        require(
            native->run(state).has_value() &&
                state[2] == static_cast<std::uint64_t>(std::int64_t{
                                std::bit_cast<std::int32_t>(word)}),
            "word shifts operate on the low word and sign extend the result");
      }
    }
  }
}

fn variable_word_shifts() -> void {
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create variable shift compiler");
  for (const std::uint32_t function : {4u, 6u, 7u}) {
    alignas(std::max_align_t) std::array<std::byte, 32768> storage;
    std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                              std::pmr::null_memory_resource()};
    const std::array words{mips::instruction_word{0x00221800u | function},
                           mips::instruction_word{0x00221000u | function}};
    let block = mips::translate(words, {0x1000}, arena);
    require(block.has_value(),
            "translate variable shifts including register overlap");
    let native = compiler->compile(*block, arena);
    require(native.has_value(), "compile variable shifts");
    for (const std::uint64_t bits :
         {0ULL, 0x80000000ULL, 0xffffffffULL, 0x1234567887654321ULL}) {
      for (const std::uint64_t count :
           {0ULL, 1ULL, 31ULL, 32ULL, 63ULL, 64ULL, 0xffffffffffffffffULL}) {
        const let amount = count & 31;
        const let low = static_cast<std::uint32_t>(bits);
        const let word = function == 4 ? low << amount
                         : function == 6
                             ? low >> amount
                             : std::bit_cast<std::uint32_t>(
                                   std::bit_cast<std::int32_t>(low) >> amount);
        const let expected = static_cast<std::uint64_t>(
            std::int64_t{std::bit_cast<std::int32_t>(word)});
        std::array<std::uint64_t, 32> state{};
        state[1] = count;
        state[2] = bits;
        require(native->run(state).has_value() && state[3] == expected &&
                    state[2] == expected,
                "variable shifts mask counts, sign extend words and preserve "
                "overlapping sources");
        let folded = ir::block::create({0x1000, 1, 32}, 3, arena);
        require(folded.has_value(), "create constant variable shift IR");
        const let input = folded->append(ir::node::constant(bits, 0));
        const let shift = folded->append(ir::node::constant(count, 0));
        require(input && shift, "append variable shift constants");
        const let value = folded->append(ir::node::binary(
            function == 4 ? ir::binary_kind::shift_left32_variable
            : function == 6
                ? ir::binary_kind::shift_right32_variable
                : ir::binary_kind::shift_right_arithmetic32_variable,
            *input, *shift, 0));
        require(value &&
                    ir::optimize(*folded, ir::fold_constants).has_value() &&
                    folded->nodes()[value->index].op == ir::opcode::constant &&
                    folded->nodes()[value->index].data.immediate == expected,
                "folded variable shifts agree with native execution");
      }
    }
    const std::array invalid{mips::instruction_word{0x00221840u | function}};
    require(!mips::translate(invalid, {0x1000}, arena),
            "variable shifts reject nonzero reserved shamt fields");
  }
}

fn doubleword_shifts() -> void {
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create doubleword shift compiler");
  for (const std::uint32_t function :
       {0x38u, 0x3au, 0x3bu, 0x3cu, 0x3eu, 0x3fu}) {
    for (const std::uint32_t field : {0u, 1u, 31u}) {
      alignas(std::max_align_t) std::array<std::byte, 32768> storage;
      std::pmr::monotonic_buffer_resource arena{
          storage.data(), storage.size(), std::pmr::null_memory_resource()};
      const std::array words{mips::instruction_word{(1u << 16) | (2u << 11) |
                                                    (field << 6) | function}};
      let block = mips::translate(words, {0x1000}, arena);
      require(block.has_value(), "translate doubleword shift");
      let native = compiler->compile(*block, arena);
      require(native.has_value(), "compile doubleword shift");
      const let amount = field + ((function & 4) ? 32 : 0);
      for (const std::uint64_t bits :
           {0ULL, 1ULL, 0x8000000000000000ULL, 0xffffffffffffffffULL,
            0x1234567887654321ULL}) {
        const let expected =
            (function & 3) == 0 ? bits << amount
            : (function & 3) == 2
                ? bits >> amount
                : std::bit_cast<std::uint64_t>(
                      std::bit_cast<std::int64_t>(bits) >> amount);
        std::array<std::uint64_t, 32> state{};
        state[1] = bits;
        require(
            native->run(state).has_value() && state[2] == expected,
            "doubleword shifts preserve all 64 bits and encoded shift counts");
        let folded = ir::block::create({0x1000, 1, 32}, 3, arena);
        require(folded.has_value(), "create constant doubleword shift IR");
        const let input = folded->append(ir::node::constant(bits, 0));
        require(input.has_value(), "append constant shift input");
        const let shifted = folded->append(
            (function & 3) == 0 ? ir::node::shift_left64(*input, amount, 0)
            : (function & 3) == 2
                ? ir::node::shift_right64(*input, amount, 0)
                : ir::node::shift_right_arithmetic64(*input, amount, 0));
        require(
            shifted && ir::optimize(*folded, ir::fold_constants).has_value() &&
                folded->nodes()[shifted->index].op == ir::opcode::constant &&
                folded->nodes()[shifted->index].data.immediate == expected,
            "constant folding agrees with doubleword shift execution");
      }
    }
  }
  alignas(std::max_align_t) std::array<std::byte, 8192> storage;
  std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                            std::pmr::null_memory_resource()};
  constexpr std::array words{mips::instruction_word{0x00221825}};
  let block = mips::translate(words, {0x1000}, arena);
  require(block.has_value(), "translate OR");
  let native = compiler->compile(*block, arena);
  require(native.has_value(), "compile OR");
  std::array<std::uint64_t, 32> state{};
  state[1] = 0xffffffff80000000ULL;
  state[2] = 0x123456789abcdef0ULL;
  require(native->run(state).has_value() && state[3] == (state[1] | state[2]),
          "OR retains both full-width operands");
}

fn word_addition() -> void {
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create word addition compiler");
  alignas(std::max_align_t) std::array<std::byte, 16384> storage;
  std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                            std::pmr::null_memory_resource()};
  for (const std::uint32_t function : {0x21u, 0x23u, 0x2au, 0x2bu}) {
    const std::array words{mips::instruction_word{0x00221800u | function}};
    let block = mips::translate(words, {0x1000}, arena);
    require(block.has_value(), "translate integer operation");
    let native = compiler->compile(*block, arena);
    require(native.has_value(), "compile integer operation");
    for (const std::uint64_t left :
         {0ULL, 0x7fffffffULL, 0xffffffffULL, 0x1234567880000000ULL,
          0xffffffffffffffffULL}) {
      for (const std::uint64_t right : {0ULL, 1ULL, 0xffffffffffffffffULL}) {
        std::array<std::uint64_t, 32> state{};
        state[1] = left;
        state[2] = right;
        const let low = static_cast<std::uint32_t>(
            function == 0x23 ? left - right : left + right);
        const let expected =
            function == 0x2a ? std::uint64_t{std::bit_cast<std::int64_t>(left) <
                                             std::bit_cast<std::int64_t>(right)}
            : function == 0x2b ? std::uint64_t{left < right}
                               : static_cast<std::uint64_t>(std::int64_t{
                                     std::bit_cast<std::int32_t>(low)});
        require(native->run(state).has_value() && state[3] == expected,
                "integer operations preserve operand order, word wrapping and "
                "signedness");
      }
    }
  }
  constexpr std::array subtract{mips::instruction_word{0x00011823}};
  let subtraction = mips::translate(subtract, {0x1000}, arena);
  require(subtraction.has_value(), "translate zero minus register");
  let negative = compiler->compile(*subtraction, arena);
  require(negative.has_value(), "compile noncommutative subtraction");
  std::array<std::uint64_t, 32> difference{};
  difference[1] = 3;
  require(negative->run(difference).has_value() &&
              difference[3] == 0xfffffffffffffffdULL,
          "SUBU cannot swap a constant left operand with its register right "
          "operand");
  constexpr std::array constants{mips::instruction_word{0x3401ffff},
                                 mips::instruction_word{0x00010fc0},
                                 mips::instruction_word{0x00211821}};
  let folded = mips::translate(constants, {0x1000}, arena);
  require(folded && ir::optimize(*folded, ir::fold_constants).has_value(),
          "fold SLL and ADDU");
  let folded_native = compiler->compile(*folded, arena);
  require(folded_native.has_value(), "compile folded word operations");
  std::array<std::uint64_t, 32> state{};
  require(folded_native->run(state).has_value() &&
              state[1] == 0xffffffff80000000ULL && state[3] == 0,
          "constant folding agrees with word shift and addition overflow");
}

fn register_nor() -> void {
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create NOR compiler");
  alignas(std::max_align_t) std::array<std::byte, 16384> storage;
  std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                            std::pmr::null_memory_resource()};
  constexpr std::array words{
      mips::instruction_word{0x00221827}, mips::instruction_word{0x00012027},
      mips::instruction_word{0x3405000f}, mips::instruction_word{0x00a02827}};
  let block = mips::translate(words, {0x1000}, arena);
  require(block && ir::optimize(*block, ir::fold_constants).has_value(),
          "translate and fold NOR through existing logical IR");
  let native = compiler->compile(*block, arena);
  require(native.has_value(), "compile NOR");
  for (const std::uint64_t left :
       {0ULL, 0x1234567887654321ULL, 0xffffffffffffffffULL}) {
    for (const std::uint64_t right : {0ULL, 0xfedcba9876543210ULL}) {
      std::array<std::uint64_t, 32> state{};
      state[1] = left;
      state[2] = right;
      require(
          native->run(state).has_value() && state[3] == ~(left | right) &&
              state[4] == ~left && state[5] == ~std::uint64_t{15},
          "NOR complements all 64 bits with register, zero and folded inputs");
    }
  }
}

fn immediate_xor() -> void {
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create XORI compiler");
  alignas(std::max_align_t) std::array<std::byte, 16384> storage;
  std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                            std::pmr::null_memory_resource()};
  constexpr std::array words{
      mips::instruction_word{0x38228000}, mips::instruction_word{0x3403ffff},
      mips::instruction_word{0x38638000}, mips::instruction_word{0x3820ffff}};
  let block = mips::translate(words, {0x1000}, arena);
  require(block && ir::optimize(*block, ir::fold_constants).has_value(),
          "translate and fold XORI");
  let native = compiler->compile(*block, arena);
  require(native.has_value(), "compile XORI");
  std::array<std::uint64_t, 32> state{};
  state[1] = 0x1234567887654321ULL;
  require(
      native->run(state).has_value() && state[0] == 0 &&
          state[2] == 0x123456788765c321ULL && state[3] == 0x7fff,
      "XORI zero extends its immediate, preserves upper bits and respects r0");
}

fn cop0_writes() -> void {
  namespace mem = jit::memory;
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create CP0 compiler");
  for (const unsigned slot : {7u, 9u, 11u, 12u, 13u, 15u, 16u, 28u, 29u}) {
    alignas(std::max_align_t) std::array<std::byte, 8192> storage;
    std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                              std::pmr::null_memory_resource()};
    const std::array words{mips::instruction_word{0x40810000u | (slot << 11)},
                           mips::instruction_word{0x40020000u | (slot << 11)}};
    let block = mips::translate(words, {0x1000}, arena);
    require(block.has_value(), "translate MTC0");
    let native = compiler->compile(*block, arena);
    require(native.has_value(), "compile MTC0");
    std::array<std::byte, mem::page_size> ram{};
    let memory = mem::address_space::create(ram, 1);
    require(memory.has_value(), "create CP0 execution memory");
    std::array<std::uint64_t, 32> state{};
    state[1] = 0xffffffff87654321ULL;
    mips::cop0_state cop0;
    cop0.registers[slot] = 0x1234;
    cop0.registers[13] = 0x8001;
    const let absent = native->run(state, **memory);
    require(!absent && absent.error().code == amd64::error_code::cop0_required,
            "CP0 instructions reject missing state");
    require(
        native->run(state, **memory, &cop0).has_value() &&
            cop0.registers[slot] ==
                (slot == 7 || slot == 15 || slot == 16 ? 0x1234u : 0x87654321u),
        "MTC0 stores low 32 bits and preserves read-only registers");
    require(state[2] == static_cast<std::uint64_t>(std::int64_t{
                            static_cast<std::int32_t>(cop0.registers[slot])}),
            "MFC0 sign extends the low word into a guest register");
    if (slot == 11)
      require(cop0.registers[13] == 1,
              "Compare write clears pending timer interrupt");
  }
}

fn cop0_full_width() -> void {
  namespace mem = jit::memory;
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create full-width CP0 compiler");
  for (const unsigned slot : {7u, 8u, 9u, 11u, 12u, 13u, 14u, 15u, 16u, 30u}) {
    alignas(std::max_align_t) std::array<std::byte, 16384> storage;
    std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                              std::pmr::null_memory_resource()};
    const std::array words{mips::instruction_word{0x40220000u | (slot << 11)},
                           mips::instruction_word{0x40030000u | (slot << 11)},
                           mips::instruction_word{0x40a10000u | (slot << 11)},
                           mips::instruction_word{0x40240000u | (slot << 11)},
                           mips::instruction_word{0x40200000u | (slot << 11)}};
    let block = mips::translate(words, {0x1000}, arena);
    require(block.has_value(), "translate full-width CP0 transfers");
    require(ir::optimize(*block, ir::fold_constants).has_value(),
            "optimize full-width CP0 transfers");
    let native = compiler->compile(*block, arena);
    require(native.has_value(), "compile full-width CP0 transfers");
    std::array<std::byte, mem::page_size> ram{};
    let memory = mem::address_space::create(ram, 1);
    require(memory.has_value(), "create full-width CP0 execution memory");
    std::array<std::uint64_t, 32> state{};
    state[1] = 0x1234567887654321ULL;
    mips::cop0_state cop0;
    cop0.registers[slot] = 0xfedcba9881234567ULL;
    require(native->run(state, **memory, &cop0).has_value(),
            "execute full-width CP0 transfers");
    for (std::size_t position = block->info().instruction_count;
         position < words.size();) {
      let tail = mips::translate(std::span{words}.subspan(position),
                                  {0x1000 + position * 4}, arena);
      require(tail.has_value(),
              "translate CP0 transfer after observation boundary");
      let code = compiler->compile(*tail, arena);
      require(code && code->run(state, **memory, &cop0).has_value(),
              "execute CP0 transfer tail");
      position += tail->info().instruction_count;
    }
    require(state[0] == 0 && state[2] == 0xfedcba9881234567ULL &&
                state[3] == 0xffffffff81234567ULL,
            "DMFC0 preserves all bits while MFC0 sign extends the low word");
    const let expected = slot == 7 || slot == 8 || slot == 15 || slot == 16
                             ? 0xfedcba9881234567ULL
                         : slot == 14 || slot == 30 ? state[1]
                                                    : 0x87654321ULL;
    require(
        state[4] == expected && cop0.registers[slot] == expected,
        "DMTC0 preserves wide exception PCs and existing register policies");
  }
  alignas(std::max_align_t) std::array<std::byte, 8192> storage;
  std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                            std::pmr::null_memory_resource()};
  for (const std::uint32_t invalid : {0x40254001u, 0x40a17001u, 0x40210800u}) {
    const std::array words{mips::instruction_word{invalid}};
    require(!mips::translate(words, {0x1000}, arena),
            "reject unsupported CP0 select and register encodings");
  }
}

fn byte_stores() -> void {
  namespace mem = jit::memory;
  std::array<std::byte, mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 1);
  require(
      memory &&
          (*memory)->map_ram({0x8000}, {0}, mem::permission::write).has_value(),
      "map byte store RAM");
  alignas(std::max_align_t) std::array<std::byte, 8192> storage;
  std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                            std::pmr::null_memory_resource()};
  constexpr std::array words{mips::instruction_word{0xa0220001}};
  let block = mips::translate(words, {0x1000}, arena);
  let compiler = amd64::compiler::create();
  require(block && compiler, "prepare SB");
  let native = compiler->compile(*block, arena);
  require(native.has_value(), "compile SB");
  std::array<std::uint64_t, 32> state{};
  state[1] = 0x8000;
  for (unsigned value = 0; value < 256; value++) {
    state[2] = 0x12340000u | value;
    require(native->run(state, **memory).has_value() &&
                ram[1] == static_cast<std::byte>(value) &&
                ram[0] == std::byte{} && ram[2] == std::byte{},
            "SB writes only its low byte at an unaligned address");
  }
  state[1] = 0x8fff;
  const let fault = native->run(state, **memory);
  require(!fault && fault.error().pc == 0x1000 &&
              fault.error().access.address.value == 0x9000,
          "SB reports an unmapped byte address");
}

fn halfword_accesses() -> void {
  namespace mem = jit::memory;
  std::array<std::byte, mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 1);
  require(memory &&
              (*memory)
                  ->map_ram({0x8000}, {0},
                            mem::permission::read | mem::permission::write)
                  .has_value(),
          "map halfword RAM");
  alignas(std::max_align_t) std::array<std::byte, 16384> storage;
  std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                            std::pmr::null_memory_resource()};
  constexpr std::array words{
      mips::instruction_word{0x34050007}, mips::instruction_word{0xa4220000},
      mips::instruction_word{0x84230000}, mips::instruction_word{0x94240000},
      mips::instruction_word{0x34050009}};
  let block = mips::translate(words, {0x1000}, arena);
  let compiler = amd64::compiler::create();
  require(block && compiler &&
              ir::optimize(*block, ir::fold_constants).has_value(),
          "translate halfword accesses");
  let native = compiler->compile(*block, arena);
  require(native.has_value(), "compile halfword accesses");
  for (const std::uint64_t value :
       {0ULL, 0x7fffULL, 0x8000ULL, 0xffffULL, 0x12345678deadbeefULL}) {
    std::array<std::uint64_t, 32> state{};
    state[1] = 0x8ffe;
    state[2] = value;
    const let low = static_cast<std::uint16_t>(value);
    require(native->run(state, **memory).has_value() &&
                state[3] == static_cast<std::uint64_t>(std::int64_t{
                                std::bit_cast<std::int16_t>(low)}) &&
                state[4] == low && state[5] == 9 &&
                ram[mem::page_size - 2] == static_cast<std::byte>(low >> 8) &&
                ram[mem::page_size - 1] == static_cast<std::byte>(low & 255),
            "SH truncates in guest byte order; LH and LHU extend at the page "
            "boundary");
  }
  for (const std::uint64_t address : {0x8fffULL, 0x9000ULL}) {
    std::array<std::uint64_t, 32> state{};
    state[1] = address;
    const let fault = native->run(state, **memory);
    require(!fault && fault.error().pc == 0x1004 && state[5] == 7 &&
                fault.error().access.address.value == address &&
                fault.error().access.size == mem::access_size::half &&
                fault.error().access.reason ==
                    (address & 1 ? mem::fault_reason::misaligned
                                 : mem::fault_reason::unmapped),
            "halfword faults preserve precise PC, width, address and earlier "
            "state");
  }
}

fn signed_byte_loads() -> void {
  namespace mem = jit::memory;
  std::array<std::byte, mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 1);
  require(
      memory &&
          (*memory)->map_ram({0x8000}, {0}, mem::permission::read).has_value(),
      "map byte load RAM");
  alignas(std::max_align_t) std::array<std::byte, 8192> storage;
  std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                            std::pmr::null_memory_resource()};
  constexpr std::array words{mips::instruction_word{0x80220001},
                             mips::instruction_word{0x90230001}};
  let block = mips::translate(words, {0x1000}, arena);
  let compiler = amd64::compiler::create();
  require(block && compiler, "prepare signed byte load");
  require(ir::optimize(*block, ir::fold_constants).has_value(),
          "optimize byte load");
  let native = compiler->compile(*block, arena);
  require(native.has_value(), "compile signed byte load");
  std::array<std::uint64_t, 32> state{};
  state[1] = 0x8000;
  for (unsigned value = 0; value < 256; value++) {
    ram[1] = static_cast<std::byte>(value);
    const let result = native->run(state, **memory);
    require(result && state[3] == value &&
                state[2] == static_cast<std::uint64_t>(
                                value < 128 ? std::int64_t{value}
                                            : std::int64_t{value} - 256),
            "LB sign extends and LBU zero extends every byte at an unaligned "
            "address");
  }
  state[1] = 0x8fff;
  const let fault = native->run(state, **memory);
  require(!fault && fault.error().pc == 0x1000 &&
              fault.error().access.address.value == 0x9000,
          "byte load reports exact unmapped address");
}

fn sign_branches() -> void {
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create sign branch compiler");
  for (std::uint32_t kind = 0; kind < 6; kind++) {
    alignas(std::max_align_t) std::array<std::byte, 8192> storage;
    std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                              std::pmr::null_memory_resource()};
    const std::array words{
        mips::instruction_word{kind < 4 ? 0x04200003u | (kind << 16)
                                        : ((kind + 2) << 26) | (1u << 21) | 3u},
        mips::instruction_word{0x34010007}, mips::instruction_word{0x34030009}};
    let block = mips::translate(words, {0x1000}, arena);
    require(block && block->info().instruction_count == 2 && block->branch(),
            "sign branches include exactly one delay slot");
    require(ir::optimize(*block, ir::fold_constants).has_value(),
            "optimize sign branch");
    let native = compiler->compile(*block, arena);
    require(native.has_value(), "compile sign branch");
    for (const std::uint64_t value :
         {0ULL, 1ULL, 0x7fffffffffffffffULL, 0x8000000000000000ULL,
          0xffffffffffffffffULL}) {
      std::array<std::uint64_t, 32> state{};
      state[1] = value;
      const bool negative = (value >> 63) != 0;
      const bool taken = kind == 4    ? negative || value == 0
                         : kind == 5  ? !negative && value != 0
                         : (kind & 1) ? !negative
                                      : negative;
      const let result = native->run(state);
      require(result && *result == (taken ? 0x1010u : 0x1008u) &&
                  state[1] ==
                      ((kind == 2 || kind == 3) && !taken ? value : 7) &&
                  state[3] == 0,
              "sign branches use full-width pre-slot values and annul likely "
              "slots");
    }
  }
}

fn branches_and_dispatch() -> void {
  namespace mem = jit::memory;
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create branch compiler");
  for (unsigned opcode = 4; opcode <= 5; opcode++) {
    for (unsigned different = 0; different < 2; different++) {
      alignas(std::max_align_t) std::array<std::byte, 8192> storage;
      std::pmr::monotonic_buffer_resource arena{
          storage.data(), storage.size(), std::pmr::null_memory_resource()};
      const std::array words{
          mips::instruction_word{(opcode << 26) | (1u << 21) | (2u << 16) | 3u},
          mips::instruction_word{0x34010007},
          mips::instruction_word{0x34030009}};
      let block = mips::translate(words, {0x1000}, arena);
      require(block && block->info().instruction_count == 2 && block->branch(),
              "branch ends block after delay slot");
      require(ir::optimize(*block, ir::fold_constants).has_value(),
              "optimize branch IR");
      let native = compiler->compile(*block, arena);
      require(native.has_value(), "compile branch");
      std::array<std::uint64_t, 32> state{};
      state[1] = 5;
      state[2] = 5 + different;
      const let result = native->run(state);
      const bool taken = opcode == 4 ? different == 0 : different != 0;
      require(result && *result == (taken ? 0x1010u : 0x1008u) &&
                  state[1] == 7 && state[3] == 0,
              "branch compares pre-slot values and executes slot once");
    }
  }
  std::array<std::byte, mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 3);
  require(memory.has_value(), "create dispatch memory");
  let &guest = **memory;
  constexpr let rw = mem::permission::read | mem::permission::write;
  constexpr let rx = mem::permission::read | mem::permission::execute;
  require(guest.map_ram({0x1000}, {0}, rw).has_value(), "map program RAM");
  constexpr std::array words{
      mips::instruction_word{0x34010000}, mips::instruction_word{0x34020003},
      mips::instruction_word{0x34040001}, mips::instruction_word{0x0024082d},
      mips::instruction_word{0x1422fffe}, mips::instruction_word{0},
      mips::instruction_word{0x34050063}};
  for (std::size_t i = 0; i < words.size(); i++)
    require(
        guest.store<std::uint32_t>({0x1000 + i * 4}, words[i].bits).has_value(),
        "load loop image");
  require(guest.protect({0x1000}, rx).has_value(), "protect code");
  require(guest.map_ram({0x2000}, {0}, rw).has_value(),
          "create writable code alias before seal");
  const let alias = jit::dispatcher::create(words, {0x1000}, guest);
  require(!alias &&
              alias.error().code == jit::dispatch_error_code::writable_code,
          "dispatcher rejects writable executable aliases");
  require(guest.unmap({0x2000}).has_value(), "remove code alias");
  let program = jit::dispatcher::create(words, {0x1000}, guest);
  require(program.has_value(), "create dispatcher");
  mips::state state{};
  state.pc = {0x1000};
  const let first = (*program)->run_slice(state, 100, 1);
  require(first && first->reason == jit::stop_reason::yielded &&
              first->instructions == 6 && first->compiled == 1 &&
              state.pc.value == 0x100c,
          "dispatch yields at block boundary");
  const let rest = (*program)->run_slice(state, 94);
  require(rest && rest->reason == jit::stop_reason::completed &&
              rest->instructions == 7 && rest->compiled == 2 &&
              state.registers[1] == 3 && state.registers[5] == 99,
          "dispatch handles backward branch and fallthrough");
  state = {};
  state.pc = {0x1000};
  const let cached = (*program)->run_slice(state, 100);
  require(cached && cached->compiled == 0 && cached->instructions == 13,
          "warm dispatch reuses compiled blocks");
  state = {};
  state.pc = {0x1000};
  const let short_budget = (*program)->run_slice(state, 5);
  require(short_budget &&
              short_budget->reason == jit::stop_reason::budget_exhausted &&
              short_budget->instructions == 0 && state.pc.value == 0x1000,
          "budget never splits a branch and delay slot");
  require(!guest.map_ram({0x2000}, {0}, rw),
          "sealed executable backing cannot gain writable alias");
  require(guest.protect({0x1000}, mem::permission::read).has_value(),
          "revoke execute access");
  const let revoked = (*program)->run_slice(state, 100);
  require(!revoked &&
              revoked.error().access.reason == mem::fault_reason::permission,
          "mapping changes invalidate cached code authorization");
  require(guest.protect({0x1000}, rx).has_value(), "restore execute access");
  const let restored = (*program)->run_slice(state, 100);
  require(restored && restored->compiled == 3,
          "mapping generation forces checked recompilation");
  require(guest.protect({0x1000}, rw).has_value(),
          "stop and replace instruction image");
  require(guest.store<std::uint32_t>({0x1000}, 0x1000ffff).has_value() &&
              guest.store<std::uint32_t>({0x1004}, 0).has_value() &&
              guest.protect({0x1000}, rx).has_value(),
          "load infinite loop");
  state = {};
  state.pc = {0x1000};
  const let changed = (*program)->run_slice(state, 100);
  require(!changed &&
              changed.error().code == jit::dispatch_error_code::code_changed,
          "changed image cannot reuse stale translations");
  constexpr std::array loop{mips::instruction_word{0x1000ffff},
                            mips::instruction_word{0}};
  let forever = jit::dispatcher::create(loop, {0x1000}, guest);
  require(forever.has_value(), "create bounded infinite loop");
  const let bounded = (*forever)->run_slice(state, 11);
  require(bounded && bounded->reason == jit::stop_reason::budget_exhausted &&
              bounded->instructions == 10 && state.pc.value == 0x1000,
          "infinite guest loop obeys instruction budget");
  alignas(std::max_align_t) std::array<std::byte, 8192> storage;
  std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                            std::pmr::null_memory_resource()};
  constexpr std::array faulting{mips::instruction_word{0x10000001},
                                mips::instruction_word{0x8c620000}};
  let block = mips::translate(faulting, {0x3000}, arena);
  require(block.has_value(), "translate faulting delay slot");
  let native = compiler->compile(*block, arena);
  require(native.has_value(), "compile faulting delay slot");
  state.registers[3] = 0x9000;
  const let failed = native->run(state.registers, guest);
  require(!failed && failed.error().pc == 0x3004 &&
              failed.error().branch_pc == 0x3000,
          "delay-slot faults retain faulting and branch PCs");
  const let missing =
      mips::translate(std::span{loop}.first(1), {0x1000}, arena);
  require(!missing &&
              missing.error().code == mips::error_code::missing_delay_slot,
          "reject missing delay slot");
  constexpr std::array nested{mips::instruction_word{0x10000000},
                              mips::instruction_word{0x10000000}};
  require(!mips::translate(nested, {0x1000}, arena),
          "reject branch in delay slot");
}

fn branch_likely() -> void {
  namespace mem = jit::memory;
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create likely branch compiler");
  for (unsigned opcode : {0x14u, 0x15u}) {
    for (unsigned different = 0; different < 2; different++) {
      for (unsigned slot = 0; slot < 4; slot++) {
        for (unsigned immediate : {1u, 3u}) {
          alignas(std::max_align_t) std::array<std::byte, 8192> storage;
          std::pmr::monotonic_buffer_resource arena{
              storage.data(), storage.size(), std::pmr::null_memory_resource()};
          const std::array words{
              mips::instruction_word{0x34030009},
              mips::instruction_word{(opcode << 26) | 0x00220000u | immediate},
              mips::instruction_word{slot == 0   ? 0x34030007u
                                     : slot == 1 ? 0xdc830000u
                                     : slot == 2 ? 0x34010007u
                                                 : 0u}};
          let block = mips::translate(words, {0x1000}, arena);
          require(block && block->branch()->annul_not_taken,
                  "translate likely branch");
          require(ir::optimize(*block, ir::fold_constants).has_value(),
                  "optimize likely branch");
          let native = compiler->compile(*block, arena);
          require(native.has_value(), "compile likely branch");
          std::array<std::byte, mem::page_size> ram{};
          let memory = mem::address_space::create(ram, 1);
          require(memory.has_value(), "create likely branch memory");
          std::array<std::uint64_t, 32> state{};
          state[1] = 5;
          state[2] = 5 + different;
          state[4] = 0x9000;
          const let initial = state;
          const let result = native->execute(state, **memory);
          const bool taken = opcode == 0x14 ? different == 0 : different != 0;
          if (slot == 1 && taken)
            require(!result && result.error().pc == 0x1008 &&
                        result.error().branch_pc == 0x1004 && state[3] == 9,
                    "taken likely slot faults with precise state");
          else
            require(result &&
                        result->next_pc ==
                            (taken ? 0x1008u + immediate * 4 : 0x100cu) &&
                        result->instructions == (taken ? 3u : 2u) &&
                        state[3] == (slot == 0 && taken ? 7u : 9u) &&
                        state[1] == (slot == 2 && taken ? 7u : 5u),
                    "likely branch annuls register writes and faults only when "
                    "not taken");
          if (slot != 1) {
            let compatibility_state = initial;
            const let compatibility = native->run(compatibility_state);
            require(compatibility && *compatibility == result->next_pc &&
                        compatibility_state == state,
                    "register-only likely branches support a null execution "
                    "context");
          }
        }
      }
    }
  }
}

fn algebraic_register_identities() -> void {
  alignas(std::max_align_t) std::array<std::byte, 16384> storage;
  std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                            std::pmr::null_memory_resource()};
  let block = ir::block::create({0x1000, 1, 32}, 16, arena);
  require(block.has_value(), "create identity IR");
  const let input = block->append(ir::node::read({1}, 0));
  const let zero = block->append(ir::node::constant(0, 0));
  const let mask = block->append(ir::node::constant(UINT64_MAX, 0));
  require(input && zero && mask, "append identity inputs");
  const let add = block->append(
      ir::node::binary(ir::binary_kind::add_wrap, *input, *zero, 0));
  const let subtract = block->append(
      ir::node::binary(ir::binary_kind::subtract_wrap, *zero, *input, 0));
  const let bit_and = block->append(
      ir::node::binary(ir::binary_kind::bit_and, *input, *mask, 0));
  const let bit_xor = block->append(
      ir::node::binary(ir::binary_kind::bit_xor, *input, *input, 0));
  require(add && subtract && bit_and && bit_xor &&
              block->append(ir::node::write({3}, *add, 0)).has_value() &&
              block->append(ir::node::write({4}, *subtract, 0)).has_value() &&
              block->append(ir::node::write({5}, *bit_and, 0)).has_value() &&
              block->append(ir::node::write({6}, *bit_xor, 0)).has_value(),
          "append identity operations");
  require(
      ir::optimize(*block, ir::fold_constants, ir::simplify_identities)
              .has_value() &&
          block->nodes()[add->index].op == ir::opcode::alias &&
          block->nodes()[bit_and->index].op == ir::opcode::alias &&
          block->nodes()[subtract->index].op == ir::opcode::subtract_wrap &&
          block->nodes()[bit_xor->index].op == ir::opcode::constant &&
          std::format("{}", *block).find("alias i64") != std::string::npos,
      "print and verify aliases while retaining noncommutative subtraction");
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create identity compiler");
  let native = compiler->compile(*block, arena);
  require(native.has_value(), "compile value aliases");
  for (const std::uint64_t bits :
       {0ULL, 1ULL, 0x8000000000000000ULL, 0x1234567887654321ULL,
        0xffffffffffffffffULL}) {
    std::array<std::uint64_t, 32> state{};
    state[1] = bits;
    require(native->run(state).has_value() && state[1] == bits &&
                state[3] == bits && state[4] == 0 - bits && state[5] == bits &&
                state[6] == 0,
            "native aliases preserve full-width SSA values and register state");
  }
}

fn architectural_tlb() -> void {
  namespace mem = jit::memory;
  namespace tlb = jit::tlb;
  for (const std::uint64_t mask : {0ULL, 0x6000ULL, 0x1e000ULL, 0x7e000ULL,
                                   0x1fe000ULL, 0x7fe000ULL, 0x1ffe000ULL}) {
    tlb::table table;
    const let bytes = ((mask | 0x1fffULL) + 1) / 2;
    require(table
                .write(5, {mask,
                           0x60000007,
                           {(0x02000000 >> 6) | 7, (0x04000000 >> 6) | 7}})
                .has_value(),
            "program paired TLB pages");
    const let even =
        table.translate({0x60000000 + bytes - 1}, 99, mem::access_kind::read);
    require(even && even->value == 0x02000000 + bytes - 1,
            "TLB even pages preserve offsets and global mappings ignore ASID");
    const let odd = table.translate({0x60000000 + bytes + 123}, 99,
                                    mem::access_kind::write);
    require(odd && odd->value == 0x04000000 + 123,
            "TLB odd page selection works from 4 KiB through 16 MiB");
  }
  tlb::table table;
  require(table.write(5, {0, 0x60000007, {6, 6}}).has_value(),
          "program ASID-local mapping");
  require(table.probe({0x60000000}, 7) == 5 && !table.probe({0x60000000}, 8),
          "local TLB entries match the current ASID");
  require(table.write(5, {0, 0x60000007, {3, 0}}).has_value(),
          "program protected and invalid halves");
  const let dirty = table.translate({0x60000000}, 7, mem::access_kind::write);
  const let invalid = table.translate({0x60001000}, 7, mem::access_kind::read);
  require(
      !dirty && dirty.error() == tlb::error::modified && !invalid &&
          invalid.error() == tlb::error::invalid &&
          !table.probe({0x60000000}, 8),
      "V/D bits and the paired global-bit rule retain distinct TLB failures");
  require(!table.write(64, {}) && !table.write(0, {0x2000, 0, {}}),
          "reject invalid TLB indices and unsupported page masks");
  alignas(std::max_align_t) std::array<std::byte, 32768> storage;
  std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                            std::pmr::null_memory_resource()};
  let compiler = amd64::compiler::create();
  std::array<std::byte, mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 1);
  require(compiler && memory, "prepare native TLB instructions");
  mips::cop0_state cp0;
  cp0.registers[0] = 5;
  cp0.registers[5] = 0x6000;
  cp0.registers[10] = 0x60000007;
  cp0.registers[2] = 6;
  cp0.registers[3] = 0x106;
  std::array<std::uint64_t, 32> state{};
  const let run = [&](std::uint32_t bits) {
    const std::array words{mips::instruction_word{bits}};
    let block = mips::translate(words, {0x1000}, arena);
    require(block.has_value(), "translate TLB operation");
    let native = compiler->compile(*block, arena);
    require(native.has_value(), "compile TLB operation");
    return native->run(state, **memory, &cp0);
  };
  require(run(0x42000002).has_value(), "TLBWI installs CP0 register contents");
  cp0.registers[5] = cp0.registers[10] = cp0.registers[2] = cp0.registers[3] =
      0;
  require(run(0x42000001).has_value() && cp0.registers[5] == 0x6000 &&
              cp0.registers[10] == 0x60000007 && cp0.registers[2] == 6 &&
              cp0.registers[3] == 0x106,
          "TLBR restores both page halves and the complete EntryHi");
  require(run(0x42000008).has_value() && cp0.registers[0] == 5,
          "TLBP reports a matching index");
  cp0.registers[10] = 0x60000008;
  require(run(0x42000008).has_value() && cp0.registers[0] == 0x80000000,
          "TLBP records probe failure when the ASID does not match");
  cp0.registers[0] = 5;
  cp0.registers[5] = 0x2000;
  const let failed = run(0x42000002);
  require(!failed &&
              failed.error().code == amd64::error_code::invalid_tlb_state &&
              failed.error().pc == 0x1000,
          "invalid native TLB writes return a typed execution failure");
}

fn prefetch_hints() -> void {
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create PREF compiler");
  for (std::uint32_t hint = 0; hint < 32; hint++) {
    alignas(std::max_align_t) std::array<std::byte, 8192> storage;
    std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                              std::pmr::null_memory_resource()};
    const std::array words{mips::instruction_word{0x34020007},
                           mips::instruction_word{0xcc200001u | (hint << 16)},
                           mips::instruction_word{0xcc20ffffu | (hint << 16)},
                           mips::instruction_word{0x34020009}};
    let block = mips::translate(words, {0x1000}, arena);
    require(block && block->info().instruction_count == 4,
            "PREF retains guest instruction accounting");
    let native = compiler->compile(*block, arena);
    require(native.has_value(), "compile optional PREF hints");
    std::array<std::uint64_t, 32> state{};
    state[1] = UINT64_MAX;
    const let result = native->run(state);
    require(
        result && *result == 0x1010 && state[1] == UINT64_MAX && state[2] == 9,
        "PREF makes no memory or MMIO access and cannot raise address faults");
  }
}

fn memory_barriers() -> void {
  namespace mem = jit::memory;
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create SYNC compiler");
  for (const std::uint32_t subtype : {0u, 1u, 31u}) {
    alignas(std::max_align_t) std::array<std::byte, 8192> storage;
    std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                              std::pmr::null_memory_resource()};
    const std::array words{mips::instruction_word{0x34010007},
                           mips::instruction_word{0x0000000fu | (subtype << 6)},
                           mips::instruction_word{0x34010009}};
    let block = mips::translate(words, {0x1000}, arena);
    require(
        block && block->info().instruction_count == 2 &&
            ir::optimize(*block, ir::fold_constants, ir::simplify_identities)
                .has_value() &&
            std::format("{}", *block).find("fence memory") != std::string::npos,
        "SYNC is a full completion barrier and explicit IR block boundary");
    let native = compiler->compile(*block, arena);
    require(native.has_value(), "compile full memory fence");
    std::array<std::uint64_t, 32> state{};
    const let result = native->run(state);
    require(result && *result == 0x1008 && state[1] == 7,
            "SYNC preserves earlier state and excludes later instructions");
    const std::array invalid{mips::instruction_word{0x0020000f}};
    require(!mips::translate(invalid, {0x1000}, arena),
            "SYNC rejects reserved register fields");
  }
  std::array<std::byte, mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 1);
  const let rwx =
      mem::permission::read | mem::permission::write | mem::permission::execute;
  require(memory && (*memory)->map_ram({0x1000}, {0}, rwx).has_value(),
          "map guest synchronization program");
  constexpr std::array words{0x34011100u, 0x340200abu, 0xac220000u,
                             0x0000000fu, 0x8c230000u, 0x70000000u};
  for (std::size_t i = 0; i < words.size(); i++)
    require(
        (*memory)->store<std::uint32_t>({0x1000 + i * 4}, words[i]).has_value(),
        "install guest synchronization program");
  let dispatch = jit::memory_dispatcher::create(**memory);
  require(dispatch.has_value(), "create synchronization dispatcher");
  mips::state state;
  state.pc = {0x1000};
  const let result = (*dispatch)->run_slice(state, 100);
  require(!result && result.error().pc == 0x1014 &&
              state.registers[3] == 0xab &&
              (*memory)->load<std::uint32_t>({0x1100}) == 0xab,
          "guest stores complete before loads across SYNC");
}

fn cp0_clock() -> void {
  mips::cop0_state cp0;
  cp0.registers[11] = 3;
  cp0.advance_cycles(1);
  require(cp0.registers[9] == 0, "Count preserves half-cycle phase");
  cp0.advance_cycles(4);
  require(cp0.registers[9] == 2 && cp0.registers[13] == 0,
          "Count advances once per two cycles");
  cp0.advance_cycles(1);
  require(cp0.registers[9] == 3 && (cp0.registers[13] & 0x8000),
          "Compare match latches timer pending");
  cp0.registers[12] = 0x8001;
  require(cp0.interrupt_pending(), "enabled timer interrupt is visible");
  cp0.registers[12] |= 2;
  require(!cp0.interrupt_pending(), "EXL suppresses interrupt entry");
  cp0.write(11, 1);
  require(!(cp0.registers[13] & 0x8000),
          "Compare write acknowledges timer interrupt");
  cp0.write(9, 0xfffffffe);
  cp0.advance_cycles(6);
  require(cp0.registers[9] == 1 && (cp0.registers[13] & 0x8000),
          "Count wraps and matches Compare across wraparound");
  cp0.write(11, 2);
  cp0.advance_cycles(std::uint64_t{1} << 33);
  require(cp0.registers[9] == 1 && (cp0.registers[13] & 0x8000),
          "full counter revolution still latches Compare");

  namespace mem = jit::memory;
  std::array<std::byte, 2 * mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 2);
  const let rwx =
      mem::permission::read | mem::permission::write | mem::permission::execute;
  require(
      memory &&
          (*memory)->map_ram({0xffffffff80000000ULL}, {0}, rwx).has_value() &&
          (*memory)
              ->map_ram({0xffffffff80001000ULL}, {mem::page_size}, rwx)
              .has_value(),
      "map clock and interrupt program");
  constexpr std::array words{0x40014800u, 0x2c220014u, 0x1440fffdu,
                             0u,          0x34030007u, 0x70000000u};
  for (std::size_t i = 0; i < words.size(); i++)
    require(
        (*memory)
            ->store<std::uint32_t>({0xffffffff80001000ULL + i * 4}, words[i])
            .has_value(),
        "install Count polling program");
  let dispatch = jit::memory_dispatcher::create(**memory);
  require(dispatch.has_value(), "create clock dispatcher");
  mips::state state;
  state.pc = {0xffffffff80001000ULL};
  const let progressed = (*dispatch)->run_slice(state, 1000, 1000);
  require(!progressed && progressed.error().pc == 0xffffffff80001014ULL &&
              state.registers[1] >= 20 && state.registers[3] == 7,
          "guest Count polling loop completes using retired block cycles");
  let immutable = mem::address_space::create(ram, 1);
  require(immutable &&
              (*immutable)
                  ->map_ram({0x1000}, {mem::page_size},
                            mem::permission::read | mem::permission::execute)
                  .has_value(),
          "map immutable Count polling image");
  std::array<mips::instruction_word, 5> polling;
  for (std::size_t i = 0; i < polling.size(); i++)
    polling[i] = {words[i]};
  let fixed = jit::dispatcher::create(polling, {0x1000}, **immutable);
  require(fixed.has_value(), "create immutable clock dispatcher");
  mips::state fixed_state;
  fixed_state.pc = {0x1000};
  const let complete = (*fixed)->run_slice(fixed_state, 1000, 1000);
  require(complete && complete->reason == jit::stop_reason::completed &&
              fixed_state.registers[1] >= 20 && fixed_state.registers[3] == 7,
          "immutable dispatcher observes Count advances between polling reads");
  require((*memory)
                  ->store<std::uint32_t>({0xffffffff80000180ULL}, 0x34040009)
                  .has_value() &&
              (*memory)
                  ->store<std::uint32_t>({0xffffffff80000184ULL}, 0x70000000)
                  .has_value(),
          "install timer exception vector");
  state = {};
  state.pc = {0xffffffff80001000ULL};
  state.cop0.registers[11] = 1;
  state.cop0.registers[12] = 0x8001;
  require((*memory)
              ->load_linked<std::uint32_t>({0xffffffff80000080ULL})
              .has_value(),
          "reserve RAM before timer interrupt");
  const let interrupted = (*dispatch)->run_slice(state, 100, 100);
  require((*memory)->store_conditional<std::uint32_t>({0xffffffff80000080ULL},
                                                      1) == false,
          "timer interrupt clears linked reservation");
  require(
      !interrupted && interrupted.error().pc == 0xffffffff80000184ULL &&
          state.registers[4] == 9 &&
          state.cop0.registers[14] == 0xffffffff80001000ULL &&
          (state.cop0.registers[12] & 2) &&
          (state.cop0.registers[13] & 0x807c) == 0x8000,
      "Compare enters guest vector with pending timer, EXL and next-block EPC");
}

fn cop1_transfers() -> void {
  namespace mem = jit::memory;
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create COP1 transfer compiler");
  std::array<std::byte, mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 0);
  require(memory.has_value(), "create COP1 execution memory");
  for (unsigned slot = 0; slot < 32; slot++) {
    alignas(std::max_align_t) std::array<std::byte, 8192> storage;
    std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                              std::pmr::null_memory_resource()};
    const std::array words{mips::instruction_word{0x44a10000u | (slot << 11)},
                           mips::instruction_word{0x44220000u | (slot << 11)},
                           mips::instruction_word{0x44a00000u | (slot << 11)},
                           mips::instruction_word{0x44230000u | (slot << 11)}};
    let block = mips::translate(words, {0x1000}, arena);
    require(block.has_value(), "translate DMTC1/DMFC1");
    let native = compiler->compile(*block, arena);
    require(native.has_value(), "compile DMTC1/DMFC1");
    std::array<std::uint64_t, 32> registers{};
    registers[1] = 0xfedcba9887654321ULL;
    const let absent = native->run(registers, **memory);
    require(!absent && absent.error().code == amd64::error_code::cop1_required,
            "COP1 transfers reject absent register state");
    mips::cop1_state cop1;
    require(
        native->run(registers, **memory, nullptr, nullptr, &cop1).has_value() &&
            registers[2] == registers[1] && registers[3] == 0 &&
            cop1.registers[slot] == 0,
        "COP1 transfers preserve all 64 bits and allow writes from r0 in every "
        "slot");
  }
}

template <bool Word> fn merged_loads() -> void {
  namespace mem = jit::memory;
  constexpr unsigned size = Word ? 4 : 8;
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create merged-load compiler");
  std::array<std::byte, 2 * mem::page_size> ram{};
  for (std::size_t i = 0; i < ram.size(); i++)
    ram[i] = static_cast<std::byte>(i & 255);
  let memory = mem::address_space::create(ram, 2);
  require(
      memory &&
          (*memory)->map_ram({0x1000}, {0}, mem::permission::read).has_value(),
      "map merged-load source");
  for (const bool right : {false, true}) {
    alignas(std::max_align_t) std::array<std::byte, 8192> storage;
    std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                              std::pmr::null_memory_resource()};
    const std::array words{
        mips::instruction_word{Word ? (right ? 0x98220000u : 0x88220000u)
                                    : (right ? 0x6c220000u : 0x68220000u)}};
    let block = mips::translate(words, {0x8000}, arena);
    require(block.has_value(), "translate word/doubleword merge loads");
    let native = compiler->compile(*block, arena);
    require(native.has_value(), "compile word/doubleword merge loads");
    for (unsigned offset = 0; offset < size; offset++) {
      std::array<std::uint64_t, 32> registers{};
      registers[1] = 0x1000 + offset;
      registers[2] = 0x8899aabbccddeeffULL;
      std::array<unsigned, 8> expected_bytes{0x88, 0x99, 0xaa, 0xbb,
                                             0xcc, 0xdd, 0xee, 0xff};
      let expected = std::span{expected_bytes}.last(size);
      if (right) {
        for (unsigned i = 0; i <= offset; i++)
          expected[size - 1 - offset + i] = i;
      } else {
        for (unsigned i = offset; i < size; i++)
          expected[i - offset] = i;
      }
      std::uint64_t value = 0;
      for (const let byte : expected)
        value = (value << 8) | byte;
      if constexpr (Word)
        value = static_cast<std::uint64_t>(std::int64_t{
            std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(value))});
      require(native->run(registers, **memory).has_value() &&
                  registers[2] == value,
              "word/doubleword merge loads merge every byte offset and "
              "preserve untouched register "
              "bytes");
    }
    std::array<std::uint64_t, 32> registers{};
    registers[1] = 0x3003;
    registers[2] = 99;
    const let fault = native->run(registers, **memory);
    require(!fault && fault.error().pc == 0x8000 &&
                fault.error().access.address.value == 0x3003 &&
                registers[2] == 99,
            "merged-load fault retains original address and destination");
  }
  alignas(std::max_align_t) std::array<std::byte, 16384> storage;
  std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                            std::pmr::null_memory_resource()};
  const std::array pair{
      mips::instruction_word{Word ? 0x88220000u : 0x68220000u},
      mips::instruction_word{Word ? 0x98220003u : 0x6c220007u}};
  let block = mips::translate(pair, {0x8000}, arena);
  require(block.has_value(), "translate paired unaligned load");
  let native = compiler->compile(*block, arena);
  require(native.has_value(), "compile paired unaligned load");
  std::array<std::uint64_t, 32> registers{};
  registers[1] = 0x1ffd;
  const let fault = native->run(registers, **memory);
  require(!fault && fault.error().pc == 0x8004 &&
              registers[2] ==
                  (Word ? 0xfffffffffdfeff00ULL : 0xfdfeff0000000000ULL),
          "faulting second merge preserves completed first merge");
  require((*memory)
              ->map_ram({0x2000}, {mem::page_size}, mem::permission::read)
              .has_value(),
          "map second source page");
  require(
      native->run(registers, **memory).has_value() &&
          registers[2] ==
              (Word ? 0xfffffffffdfeff00ULL : 0xfdfeff0001020304ULL),
      "word/doubleword merge loads pair joins bytes across a page boundary");
}

template <bool Doubleword> fn merged_stores() -> void {
  namespace mem = jit::memory;
  constexpr unsigned size = Doubleword ? 8 : 4;
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create merged-store compiler");
  std::array<std::byte, 2 * mem::page_size> ram;
  let memory = mem::address_space::create(ram, 3);
  require(
      memory &&
          (*memory)->map_ram({0x1000}, {0}, mem::permission::write).has_value(),
      "map merged-store RAM");
  unsigned callbacks = 0;
  mem::device endpoint{
      &callbacks,
      [](void *context, mem::device_access) noexcept
          -> std::expected<std::uint64_t, mem::device_error> {
        (*static_cast<unsigned *>(context))++;
        return 0;
      },
      [](void *context, mem::device_access,
         std::uint64_t) noexcept -> std::expected<void, mem::device_error> {
        (*static_cast<unsigned *>(context))++;
        return {};
      }};
  require(
      (*memory)
          ->map_device({0x4000}, {0x100000}, endpoint, mem::permission::write)
          .has_value(),
      "map partial-store device");
  constexpr std::array source_bytes{
      std::byte{0x12}, std::byte{0x34}, std::byte{0x56}, std::byte{0x78},
      std::byte{0x89}, std::byte{0xab}, std::byte{0xcd}, std::byte{0xef}};
  const let source = std::span{source_bytes}.last(size);
  for (const bool right : {false, true}) {
    alignas(std::max_align_t) std::array<std::byte, 8192> storage;
    std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                              std::pmr::null_memory_resource()};
    const std::array words{mips::instruction_word{
        Doubleword ? (right ? 0xb4220000u : 0xb0220000u)
                   : (right ? 0xb8220000u : 0xa8220000u)}};
    let block = mips::translate(words, {0x8000}, arena);
    require(block.has_value(), "translate word/doubleword merge stores");
    let native = compiler->compile(*block, arena);
    require(native.has_value(), "compile word/doubleword merge stores");
    for (unsigned offset = 0; offset < size; offset++) {
      ram.fill(std::byte{0xaa});
      std::array<std::byte, 16> expected;
      expected.fill(std::byte{0xaa});
      if (right)
        for (unsigned i = 0; i <= offset; i++)
          expected[i] = source[size - 1 - offset + i];
      else
        for (unsigned i = offset; i < size; i++)
          expected[i] = source[i - offset];
      std::array<std::uint64_t, 32> registers{};
      registers[1] = 0x1000 + offset;
      registers[2] = 0x1234567889abcdefULL;
      require(native->run(registers, **memory).has_value() &&
                  std::ranges::equal(expected, std::span{ram}.first(16)),
              "word/doubleword merge stores write only selected bytes at every "
              "offset without read "
              "permission");
    }
    std::array<std::uint64_t, 32> registers{};
    registers[1] = 0x3003;
    registers[2] = 0x1234567889abcdefULL;
    const let before = ram;
    const let absent = native->run(registers, **memory);
    require(!absent && absent.error().pc == 0x8000 &&
                absent.error().access.address.value == 0x3003 && ram == before,
            "unmapped partial store retains its original address and has no "
            "effect");
    registers[1] = 0x4001;
    const let device = native->run(registers, **memory);
    require(!device &&
                device.error().access.reason ==
                    mem::fault_reason::unsupported_size &&
                callbacks == 0,
            "partial MMIO stores fail without any device read or write");
  }
  alignas(std::max_align_t) std::array<std::byte, 16384> storage;
  std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                            std::pmr::null_memory_resource()};
  const std::array pair{
      mips::instruction_word{Doubleword ? 0xb0220000u : 0xa8220000u},
      mips::instruction_word{Doubleword ? 0xb4220007u : 0xb8220003u}};
  let block = mips::translate(pair, {0x8000}, arena);
  require(block.has_value(), "translate paired partial stores");
  let native = compiler->compile(*block, arena);
  require(native.has_value(), "compile paired partial stores");
  ram.fill(std::byte{0xaa});
  std::array<std::uint64_t, 32> registers{};
  registers[1] = 0x1ffd;
  registers[2] = 0x1234567889abcdefULL;
  const let fault = native->run(registers, **memory);
  require(!fault && fault.error().pc == 0x8004 &&
              ram[mem::page_size - 3] == source[0] &&
              ram[mem::page_size - 1] == source[2] &&
              ram[mem::page_size] == std::byte{0xaa},
          "faulting second partial store preserves the first store only");
  require((*memory)
              ->map_ram({0x2000}, {mem::page_size}, mem::permission::write)
              .has_value(),
          "map second partial-store page");
  require(native->run(registers, **memory).has_value() &&
              std::ranges::equal(
                  source, std::span{ram}.subspan(mem::page_size - 3, size)),
          "word/doubleword merge stores pair stores an unaligned word across "
          "pages");
}

fn cop1_memory_transfers() -> void {
  namespace mem = jit::memory;
  let compiler = amd64::compiler::create();
  std::array<std::byte, mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 1);
  require(compiler && memory &&
              (*memory)
                  ->map_ram({0x1000}, {0},
                            mem::permission::read | mem::permission::write)
                  .has_value(),
          "prepare COP1 memory transfers");
  alignas(std::max_align_t) std::array<std::byte, 16384> storage;
  std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                            std::pmr::null_memory_resource()};
  const std::array words{mips::instruction_word{0xf4200000},
                         mips::instruction_word{0xd43f0008},
                         mips::instruction_word{0x4422f800}};
  let block = mips::translate(words, {0x8000}, arena);
  require(block.has_value(), "translate SDC1, LDC1 and DMFC1");
  let native = compiler->compile(*block, arena);
  require(native.has_value(), "compile COP1 memory transfers");
  std::array<std::uint64_t, 32> registers{};
  registers[1] = 0x1000;
  mips::cop1_state cop1;
  cop1.registers[0] = 0x7ff8123487654321ULL;
  require((*memory)
              ->store<std::uint64_t>({0x1008}, 0xfedcba9876543210ULL)
              .has_value(),
          "seed COP1 load bits");
  require(
      native->run(registers, **memory, nullptr, nullptr, &cop1).has_value() &&
          (*memory)->load<std::uint64_t>({0x1000}) == cop1.registers[0] &&
          cop1.registers[31] == 0xfedcba9876543210ULL &&
          registers[2] == cop1.registers[31],
      "COP1 memory transfers preserve every bit in f0 and f31");
  const let before = ram;
  registers[1] = 0x1001;
  const let misaligned =
      native->run(registers, **memory, nullptr, nullptr, &cop1);
  require(!misaligned && misaligned.error().pc == 0x8000 &&
              misaligned.error().access.reason ==
                  mem::fault_reason::misaligned &&
              ram == before,
          "misaligned COP1 store has no partial effect");
  registers[1] = 0x1ff8;
  cop1.registers[31] = 99;
  const let absent = native->run(registers, **memory, nullptr, nullptr, &cop1);
  require(
      !absent && absent.error().pc == 0x8004 && cop1.registers[31] == 99 &&
          (*memory)->load<std::uint64_t>({0x1ff8}) == cop1.registers[0],
      "faulting COP1 load preserves its destination and the preceding store");
}

fn linked_word_operations() -> void {
  namespace mem = jit::memory;
  let compiler = amd64::compiler::create();
  std::array<std::byte, mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 1);
  require(compiler && memory &&
              (*memory)
                  ->map_ram({0x1000}, {0},
                            mem::permission::read | mem::permission::write)
                  .has_value(),
          "prepare LL/SC execution");
  for (unsigned scenario = 0; scenario < 3; scenario++) {
    alignas(std::max_align_t) std::array<std::byte, 16384> storage;
    std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                              std::pmr::null_memory_resource()};
    const std::array words{
        mips::instruction_word{0xc0220000},
        mips::instruction_word{scenario == 1 ? 0xac200004u : 0x24420001u},
        mips::instruction_word{scenario == 2 ? 0xe0200000u : 0xe0220000u},
        mips::instruction_word{0x34430000}};
    let block = mips::translate(words, {0x8000}, arena);
    require(block.has_value(), "translate LL/add/SC");
    let native = compiler->compile(*block, arena);
    require(native.has_value(), "compile LL/add/SC");
    require((*memory)->store<std::uint32_t>({0x1000}, 0x7fffffff).has_value(),
            "seed LL/SC target");
    std::array<std::uint64_t, 32> registers{};
    registers[1] = 0x1000;
    require(native->run(registers, **memory).has_value() && registers[0] == 0 &&
                registers[2] == (scenario == 2   ? 0xffffffff80000000ULL
                                 : scenario == 1 ? 0
                                                 : 1) &&
                (*memory)->load<std::uint32_t>({0x1000}) == (scenario == 2 ? 0u
                                                             : scenario == 1
                                                                 ? 0x7fffffffu
                                                                 : 0x80000000u),
            "SC reports success, rejects intervening stores, and retains r0 "
            "effects");
  }
  for (const std::uint32_t operation :
       {0x0000000cu, 0x42000018u, 0x20210001u}) {
    alignas(std::max_align_t) std::array<std::byte, 8192> storage;
    std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                              std::pmr::null_memory_resource()};
    const std::array words{mips::instruction_word{operation}};
    let block = mips::translate(words, {0x8000}, arena);
    require(block.has_value(), "translate reservation exception boundary");
    let native = compiler->compile(*block, arena);
    require(native.has_value(), "compile reservation exception boundary");
    std::array<std::uint64_t, 32> registers{};
    registers[1] = 0x7fffffff;
    mips::cop0_state cp0;
    require((*memory)->load_linked<std::uint32_t>({0x1000}).has_value() &&
                native->run(registers, **memory, &cp0).has_value() &&
                (*memory)->store_conditional<std::uint32_t>({0x1000}, 1) ==
                    false,
            "SYSCALL, ERET and overflow invalidate linked reservations");
  }
}

fn unsigned_word_load() -> void {
  namespace mem = jit::memory;
  let compiler = amd64::compiler::create();
  std::array<std::byte, mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 1);
  require(compiler && memory &&
              (*memory)
                  ->map_ram({0x1000}, {0},
                            mem::permission::read | mem::permission::write)
                  .has_value(),
          "prepare unsigned word loads");
  for (const unsigned destination : {0u, 2u}) {
    alignas(std::max_align_t) std::array<std::byte, 8192> storage;
    std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                              std::pmr::null_memory_resource()};
    const std::array words{
        mips::instruction_word{0x9c200000u | (destination << 16)},
        mips::instruction_word{0x8c230000}};
    let block = mips::translate(words, {0x8000}, arena);
    require(block.has_value(), "translate LWU and LW");
    let native = compiler->compile(*block, arena);
    require(native.has_value(), "compile LWU and LW");
    for (const std::uint32_t value :
         {0u, 0x7fffffffu, 0x80000000u, 0xffffffffu}) {
      require((*memory)->store<std::uint32_t>({0x1000}, value).has_value(),
              "seed unsigned word source");
      std::array<std::uint64_t, 32> registers{};
      registers[1] = 0x1000;
      require(native->run(registers, **memory).has_value() &&
                  registers[0] == 0 &&
                  registers[2] == (destination ? value : 0) &&
                  registers[3] == static_cast<std::uint64_t>(std::int64_t{
                                      std::bit_cast<std::int32_t>(value)}),
              "LWU zero extends while LW sign extends bit 31");
    }
    std::array<std::uint64_t, 32> registers{};
    registers[1] = 0x1001;
    const let fault = native->run(registers, **memory);
    require(!fault && fault.error().pc == 0x8000 &&
                fault.error().access.reason == mem::fault_reason::misaligned,
            "LWU validates alignment even when its destination is r0");
  }
}

fn indexed_cache_maintenance() -> void {
  namespace mem = jit::memory;
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create cache maintenance compiler");
  for (const std::uint32_t operation : {0u, 1u, 11u, 22u}) {
    alignas(std::max_align_t) std::array<std::byte, 8192> storage;
    std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                              std::pmr::null_memory_resource()};
    const std::array words{
        mips::instruction_word{0x34020007},
        mips::instruction_word{0xbc20ffffu | (operation << 16)},
        mips::instruction_word{0x34020009}};
    let block = mips::translate(words, {0x1000}, arena);
    require(block && block->info().instruction_count == 2 &&
                ir::optimize(*block, ir::fold_constants).has_value(),
            "indexed cache maintenance forms an explicit block boundary");
    let native = compiler->compile(*block, arena);
    require(native.has_value(), "compile indexed cache maintenance");
    std::array<std::uint64_t, 32> state{};
    std::array<std::byte, mem::page_size> ram{};
    let memory = mem::address_space::create(ram, 2);
    require(memory.has_value(), "create cache address translation space");
    state[1] = 0xffffffff80001000ULL;
    const let result = native->run(state, **memory);
    require(result && *result == 0x1008 && state[2] == 7 &&
                state[1] == 0xffffffff80001000ULL,
            "indexed maintenance requires no mapped address and preserves "
            "registers");
    state[1] = 0x4000;
    const let invalid = native->run(state, **memory);
    require(!invalid && invalid.error().pc == 0x1004 &&
                invalid.error().access.address.value == 0x3fff &&
                invalid.error().access.reason == mem::fault_reason::unmapped &&
                state[2] == 7,
            "mapped cache indices require translation and retain precise fault "
            "state");
    require(
        (*memory)->map_ram({0x8000}, {0}, mem::permission::read).has_value(),
        "map a valid cache index address");
    state[1] = 0x8001;
    require(native->run(state, **memory).has_value(),
            "mapped cache indices translate without data access");
    unsigned reads = 0;
    mem::device endpoint{
        &reads,
        [](void *context, mem::device_access) noexcept
            -> std::expected<std::uint64_t, mem::device_error> {
          (*static_cast<unsigned *>(context))++;
          return 0;
        },
        nullptr};
    require(
        (*memory)
            ->map_device({0x9000}, {0x100000}, endpoint, mem::permission::read)
            .has_value(),
        "map cache index device translation");
    state[1] = 0x9001;
    require(native->run(state, **memory).has_value() && reads == 0,
            "cache translation never performs MMIO register reads");
    const std::array unsupported{mips::instruction_word{0xbd100000}};
    require(!mips::translate(unsupported, {0x1000}, arena),
            "hit and tag cache operations require their own modeled semantics");
  }
}

fn hilo_arithmetic() -> void {
  namespace mem = jit::memory;

  struct sample {
    std::uint64_t left, right, high, low;
  };

  constexpr std::array multiplication{
      sample{0, 0xffffffff, 0, 0},
      sample{0x7fffffff, 2, 0, 0xfffffffffffffffeULL},
      sample{0x80000000, 0xffffffff, 0, 0xffffffff80000000ULL},
      sample{0x80000000, 2, UINT64_MAX, 0},
      sample{0x1234567800000003ULL, 0xfedcba9800000007ULL, 0, 21}};
  constexpr std::array unsigned_multiplication{
      sample{0, 0xffffffff, 0, 0},
      sample{0xffffffff, 0xffffffff, 0xfffffffffffffffeULL, 1},
      sample{0xffffffff, 2, 1, 0xfffffffffffffffeULL},
      sample{0x80000000, 2, 1, 0},
      sample{0x1234567800000003ULL, 0xfedcba9800000007ULL, 0, 21}};
  constexpr std::array signed_division{
      sample{7, 3, 1, 2},
      sample{0xfffffff9, 3, UINT64_MAX, 0xfffffffffffffffeULL},
      sample{7, 0xfffffffd, 1, 0xfffffffffffffffeULL},
      sample{0x80000000, 0xffffffff, 0, 0xffffffff80000000ULL},
      sample{0x80000000, 1, 0, 0xffffffff80000000ULL},
      sample{7, 0, 0xaabb, 0xccdd}};
  constexpr std::array division{
      sample{0xffffffff, 2, 1, 0x7fffffff},
      sample{0x80000000, 1, 0, 0xffffffff80000000ULL},
      sample{7, 3, 1, 2},
      sample{0x1234567800000007ULL, 0xfedcba9800000003ULL, 1, 2},
      sample{0, 3, 0, 0},
      sample{7, 0, 0xaabb, 0xccdd},
      sample{7, 0x100000000ULL, 0xaabb, 0xccdd}};
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create HI/LO arithmetic compiler");
  std::array<std::byte, mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 1);
  require(memory.has_value(), "create HI/LO arithmetic memory");
  for (const std::uint32_t function : {0x18u, 0x19u, 0x1au, 0x1bu}) {
    alignas(std::max_align_t) std::array<std::byte, 16384> storage;
    std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                              std::pmr::null_memory_resource()};
    const std::array words{mips::instruction_word{0x00220000u | function},
                           mips::instruction_word{0x00001810},
                           mips::instruction_word{0x00002012}};
    let block = mips::translate(words, {0x1000}, arena);
    require(block && ir::optimize(*block, ir::fold_constants).has_value(),
            "translate and optimize HI/LO arithmetic effects");
    let native = compiler->compile(*block, arena);
    require(native.has_value(), "compile HI/LO arithmetic");
    const std::span<const sample> samples =
        function == 0x18   ? std::span<const sample>{multiplication}
        : function == 0x19 ? std::span<const sample>{unsigned_multiplication}
        : function == 0x1a ? std::span<const sample>{signed_division}
                           : std::span<const sample>{division};
    for (const let &test : samples) {
      std::array<std::uint64_t, 32> state{};
      state[1] = test.left;
      state[2] = test.right;
      mips::hilo_state hilo;
      hilo.registers = {0xaabb, 0xccdd};
      const let absent = native->run(state, **memory);
      require(!absent &&
                  absent.error().code == amd64::error_code::hilo_required,
              "HI/LO arithmetic requires its state bank");
      require(native->run(state, **memory, nullptr, &hilo).has_value() &&
                  hilo.registers[0] == test.high &&
                  hilo.registers[1] == test.low && state[3] == test.high &&
                  state[4] == test.low && state[1] == test.left &&
                  state[2] == test.right,
              "MULT and DIVU produce sign-extended word halves and preserve "
              "zero-divisor state");
    }
    const std::array invalid{mips::instruction_word{0x00220800u | function}};
    require(!mips::translate(invalid, {0x1000}, arena),
            "HI/LO arithmetic rejects reserved destination fields");
  }
}

fn hilo_transfers() -> void {
  namespace mem = jit::memory;
  alignas(std::max_align_t) std::array<std::byte, 16384> storage;
  std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                            std::pmr::null_memory_resource()};
  constexpr std::array words{
      mips::instruction_word{0x00200011}, mips::instruction_word{0x00400013},
      mips::instruction_word{0x00001810}, mips::instruction_word{0x00002012},
      mips::instruction_word{0x00000011}, mips::instruction_word{0x00000013},
      mips::instruction_word{0x00002810}, mips::instruction_word{0x00003012}};
  let block = mips::translate(words, {0x1000}, arena);
  let compiler = amd64::compiler::create();
  require(block && compiler, "prepare HI/LO transfers");
  require(ir::optimize(*block, ir::fold_constants).has_value(),
          "optimize HI/LO transfers");
  let native = compiler->compile(*block, arena);
  require(native.has_value(), "compile HI/LO transfers");
  std::array<std::byte, mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 1);
  require(memory.has_value(), "create HI/LO execution memory");
  std::array<std::uint64_t, 32> state{};
  state[1] = 0xfedcba9876543210ULL;
  state[2] = 0x0123456789abcdefULL;
  mips::hilo_state hilo{};
  const let absent = native->run(state, **memory);
  require(!absent && absent.error().code == amd64::error_code::hilo_required,
          "reject missing HI/LO state");
  require(
      native->run(state, **memory, nullptr, &hilo).has_value() &&
          state[3] == state[1] && state[4] == state[2] && state[5] == 0 &&
          state[6] == 0 && hilo.registers[0] == 0 && hilo.registers[1] == 0,
      "HI/LO transfers retain full-width values and accept the zero register");
  for (const std::uint32_t invalid : {0x00201810u, 0x00210011u}) {
    const std::array bad{mips::instruction_word{invalid}};
    const let rejected = mips::translate(bad, {0x1000}, arena);
    require(!rejected &&
                rejected.error().code == mips::error_code::invalid_encoding,
            "HI/LO transfers validate reserved register fields");
  }
}

fn checked_addition() -> void {
  namespace mem = jit::memory;
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create checked addition compiler");
  std::array<std::byte, mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 1);
  require(memory.has_value(), "create checked addition memory");
  constexpr std::uint64_t start = 0xffffffff80001000ULL;
  for (unsigned slot = 0; slot < 3; slot++) {
    for (const std::uint32_t destination : {0u, 2u}) {
      for (const std::uint16_t immediate :
           {std::uint16_t{1}, std::uint16_t{0xffff}}) {
        alignas(std::max_align_t) std::array<std::byte, 16384> storage;
        std::pmr::monotonic_buffer_resource arena{
            storage.data(), storage.size(), std::pmr::null_memory_resource()};
        const mips::instruction_word add{0x20200000u | (destination << 16) |
                                         immediate};
        const std::array words{
            mips::instruction_word{0x34030007},
            slot ? mips::instruction_word{slot == 1 ? 0x10000001u : 0x54000001u}
                 : add,
            slot ? add : mips::instruction_word{0x34030009},
            mips::instruction_word{0x34030009}};
        let block = mips::translate(words, {start}, arena);
        require(block && ir::optimize(*block, ir::fold_constants).has_value(),
                "translate and optimize checked addition");
        let native = compiler->compile(*block, arena);
        require(native.has_value(), "compile checked addition");
        for (const std::uint64_t bits :
             {0ULL, 0x7fffffffULL, 0x80000000ULL, 0xffffffffULL,
              0x1234567880000000ULL}) {
          const let sum = std::int64_t{std::bit_cast<std::int32_t>(
                              static_cast<std::uint32_t>(bits))} +
                          std::bit_cast<std::int16_t>(immediate);
          const bool overflow =
              slot != 2 && (sum > 0x7fffffffLL || sum < -0x80000000LL);
          std::array<std::uint64_t, 32> state{};
          state[1] = bits;
          state[2] = 99;
          mips::cop0_state cop0;
          cop0.registers[13] = 0x8000;
          const let result = native->execute(state, **memory, &cop0);
          require(result && result->next_pc == (overflow ? 0xffffffff80000180ULL
                                                         : block->next_pc()),
                  "ADDI returns an overflow vector or the ordinary next PC");
          require(result->reason == (overflow
                                         ? amd64::exit_reason::guest_exception
                                         : amd64::exit_reason::completed) &&
                      result->instructions ==
                          (overflow ? (slot == 1 ? 3u : 2u)
                                    : block->info().instruction_count -
                                          (slot == 2 ? 1u : 0u)),
                  "native outcome counts the precise prefix, including "
                  "delay-slot traps");
          require(state[0] == 0 &&
                      state[2] == (overflow || slot == 2 || destination == 0
                                       ? 99ULL
                                       : static_cast<std::uint64_t>(sum)) &&
                      state[3] == (overflow || slot ? 7u : 9u),
                  "ADDI preserves precise state and traps even when its "
                  "destination is r0");
          require(cop0.registers[13] ==
                          (overflow
                               ? 0x8030ULL | (slot == 1 ? 0x80000000ULL : 0)
                               : 0x8000ULL) &&
                      cop0.registers[12] == (overflow ? 2u : 0u) &&
                      cop0.registers[14] == (overflow ? start + 4 : 0),
                  "ADDI overflow sets ExcCode, EXL, EPC and delay-slot BD "
                  "precisely");
        }
      }
    }
  }
}

fn exception_return() -> void {
  namespace mem = jit::memory;
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create exception return compiler");
  alignas(std::max_align_t) std::array<std::byte, 16384> storage;
  std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                            std::pmr::null_memory_resource()};
  constexpr std::array words{mips::instruction_word{0x34010007},
                             mips::instruction_word{0x42000018},
                             mips::instruction_word{0x34010009}};
  let block = mips::translate(words, {0x1000}, arena);
  require(block && block->info().instruction_count == 2 &&
              ir::optimize(*block, ir::fold_constants).has_value(),
          "ERET ends the translated block without a delay slot");
  let native = compiler->compile(*block, arena);
  require(native.has_value(), "compile ERET");
  std::array<std::byte, mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 1);
  require(memory.has_value(), "create ERET execution memory");
  for (const std::uint64_t level : {0ULL, 2ULL, 4ULL, 6ULL}) {
    std::array<std::uint64_t, 32> state{};
    const let absent = native->run(state, **memory);
    require(!absent && absent.error().code == amd64::error_code::cop0_required,
            "ERET rejects missing CP0 state");
    mips::cop0_state cop0;
    cop0.registers[12] = 0x10400001 | level;
    cop0.registers[13] = 0x80008020;
    cop0.registers[14] = 0xffffffff800080d4ULL;
    cop0.registers[30] = 0xffffffff81234560ULL;
    const let result = native->run(state, **memory, &cop0);
    require(result && *result == cop0.registers[level & 4 ? 30 : 14] &&
                cop0.registers[12] ==
                    ((0x10400001 | level) & ~(level & 4 ? 4ULL : 2ULL)) &&
                cop0.registers[13] == 0x80008020 && state[1] == 7,
            "ERET selects full-width EPC or ErrorEPC and clears only its "
            "exception level");
  }
  constexpr std::array invalid{mips::instruction_word{0x10000001},
                               mips::instruction_word{0x42000018}};
  const let rejected = mips::translate(invalid, {0x1000}, arena);
  require(!rejected &&
              rejected.error().code == mips::error_code::invalid_encoding &&
              rejected.error().instruction_index == 1,
          "reject ERET in a branch delay slot");
}

fn syscall_exceptions() -> void {
  namespace mem = jit::memory;
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create syscall compiler");
  constexpr std::uint64_t pc = 0xffffffff80001000ULL;
  for (unsigned bev = 0; bev < 2; bev++) {
    for (unsigned scenario = 0; scenario < 4; scenario++) {
      alignas(std::max_align_t) std::array<std::byte, 8192> storage;
      std::pmr::monotonic_buffer_resource arena{
          storage.data(), storage.size(), std::pmr::null_memory_resource()};
      const bool delay = scenario == 1;
      const bool annulled = scenario == 2;
      const bool nested = scenario == 3;
      const std::array words{mips::instruction_word{delay      ? 0x10000001u
                                                    : annulled ? 0x54220001u
                                                               : 0x34030009u},
                             mips::instruction_word{(0x12345u << 6) | 0x0cu},
                             mips::instruction_word{0x34040007}};
      let block = mips::translate(words, {pc}, arena);
      require(block && block->info().instruction_count == 2,
              "syscall terminates its translated block");
      let native = compiler->compile(*block, arena);
      require(native.has_value(), "compile syscall");
      std::array<std::byte, mem::page_size> ram{};
      let memory = mem::address_space::create(ram, 1);
      require(memory.has_value(), "create syscall memory");
      std::array<std::uint64_t, 32> state{};
      state[1] = state[2] = 5;
      mips::cop0_state cop0;
      cop0.registers[12] = (bev ? 1u << 22 : 0) | (nested ? 2u : 0);
      cop0.registers[13] = 0x80008000;
      cop0.registers[14] = 0xffffffff81234560ULL;
      const let result = native->run(state, **memory, &cop0);
      require(result.has_value(), "execute syscall exception entry");
      if (annulled) {
        require(*result == pc + 8 &&
                    cop0.registers[14] == 0xffffffff81234560ULL &&
                    !(cop0.registers[12] & 2),
                "annulled likely slot cannot raise a syscall");
      } else {
        require(*result ==
                        (bev ? 0xffffffffbfc00380ULL : 0xffffffff80000180ULL) &&
                    (cop0.registers[12] & 2) &&
                    (cop0.registers[13] & 0x7c) == 0x20 &&
                    (cop0.registers[13] & 0x8000) && state[4] == 0 &&
                    cop0.registers[14] == (nested  ? 0xffffffff81234560ULL
                                           : delay ? pc
                                                   : pc + 4),
                "syscall selects the vector, preserves pending IRQs and "
                "records precise EPC");
        require(
            ((cop0.registers[13] >> 31) & 1) == (delay || nested ? 1u : 0u),
            "syscall records delay slots and preserves nested EPC/BD state");
        if (!delay)
          require(state[3] == 9,
                  "architectural writes before the syscall are committed");
      }
    }
  }
}

fn writable_code_dispatch() -> void {
  namespace mem = jit::memory;
  std::array<std::byte, mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 2);
  const let rwx =
      mem::permission::read | mem::permission::write | mem::permission::execute;
  require(memory && (*memory)->map_ram({0x1000}, {0}, rwx).has_value() &&
              (*memory)->map_ram({0x3000}, {0}, rwx).has_value(),
          "map writable code aliases");
  require(
      (*memory)->store<std::uint32_t>({0x1080}, 0x34040001).has_value() &&
          (*memory)->store<std::uint32_t>({0x1084}, 0x20850001).has_value() &&
          (*memory)->store<std::uint32_t>({0x1088}, 0x34060009).has_value() &&
          (*memory)->store<std::uint32_t>({0x108c}, 0x70000000).has_value(),
      "install bounded execution program");
  let bounded = jit::memory_dispatcher::create(**memory);
  require(bounded.has_value(), "create bounded dispatcher");
  mips::state precise{};
  precise.pc = {0x1080};
  const let warm = (*bounded)->run_slice(precise, 3, 1);
  require(warm && warm->instructions == 3, "warm a larger cached block");
  precise = {};
  precise.pc = {0x1080};
  const let tiny = (*bounded)->run_slice(precise, 1);
  require(tiny && tiny->instructions == 1 && precise.pc.value == 0x1084 &&
              precise.registers[4] == 1,
          "small budgets execute a prefix rather than stall");
  precise.pc = {0x1080};
  require((*memory)->store<std::uint32_t>({0x1084}, 0x20e50001).has_value(),
          "install overflowing ADDI");
  precise.registers[7] = 0x7fffffff;
  bounded = jit::memory_dispatcher::create(**memory);
  require(bounded.has_value(), "create full block trap dispatcher");
  const let trapped = (*bounded)->run_slice(precise, 10, 1);
  require(trapped && trapped->reason == jit::stop_reason::guest_exception &&
              trapped->instructions == 2 &&
              precise.pc.value == 0xffffffff80000180ULL &&
              precise.cop0.registers[9] == 1 && precise.cop0.half_cycle &&
              precise.registers[6] == 0,
          "overflow charges the executed prefix including the trapping slot");
  constexpr std::array words{0x34011020u, 0x3c023403u, 0x34420007u, 0xac220000u,
                             0x08000408u, 0u,          0x70000000u, 0u,
                             0x34030001u, 0x70000000u};
  for (std::size_t i = 0; i < words.size(); i++)
    require(
        (*memory)->store<std::uint32_t>({0x1000 + i * 4}, words[i]).has_value(),
        "load mutable program");
  let dispatch = jit::memory_dispatcher::create(**memory);
  require(dispatch.has_value(), "create memory dispatcher");
  mips::state state{};
  state.pc = {0x1020};
  const let first = (*dispatch)->run_slice(state, 1);
  require(first && state.registers[3] == 1 && state.pc.value == 0x1024,
          "execute the valid prefix before an unsupported instruction");
  state.pc = {0x1000};
  const let changed = (*dispatch)->run_slice(state, 100);
  require(!changed &&
              changed.error().code == jit::dispatch_error_code::translation &&
              changed.error().pc == 0x1024 && state.registers[3] == 7,
          "guest store updates cached code before control reaches it");
  require((*memory)->store<std::uint32_t>({0x3020}, 0x34030009).has_value(),
          "replace code through its physical alias");
  state.pc = {0x1020};
  const let alias = (*dispatch)->run_slice(state, 1);
  require(alias && alias->compiled == 1 && state.registers[3] == 9,
          "alias writes invalidate cached instruction bytes");
  state.pc = {0x1020};
  const let cached = (*dispatch)->run_slice(state, 1);
  require(cached && cached->compiled == 0 && state.registers[3] == 9,
          "unchanged code reuses its native block");
  (*memory)->invalidate();
  state.pc = {0x1020};
  const let revalidated = (*dispatch)->run_slice(state, 1);
  require(revalidated && revalidated->compiled == 0 && state.registers[3] == 9,
          "translation invalidation revalidates backing without recompilation");
  constexpr std::array maintenance{0xbd210001u, 0xbd200000u, 0x08000408u, 0u};
  for (std::size_t i = 0; i < maintenance.size(); i++)
    require((*memory)
                ->store<std::uint32_t>({0x1040 + i * 4}, maintenance[i])
                .has_value(),
            "install indexed cache maintenance sequence");
  require((*memory)->store<std::uint32_t>({0x3020}, 0x3403000b).has_value(),
          "modify warmed code through an alias before cache maintenance");
  state.pc = {0x1040};
  state.registers[9] = 0xffffffff80000000ULL;
  const let maintained = (*dispatch)->run_slice(state, 100);
  require(!maintained && maintained.error().pc == 0x1024 &&
              state.registers[3] == 11,
          "D writeback and I invalidation publish modified code without "
          "discarding RAM");
  require(
      (*memory)
          ->protect({0x1000}, mem::permission::read | mem::permission::write)
          .has_value(),
      "revoke code execution");
  state.pc = {0x1020};
  const let denied = (*dispatch)->run_slice(state, 1);
  require(!denied && denied.error().code == jit::dispatch_error_code::memory &&
              denied.error().access.reason == mem::fault_reason::permission,
          "cached code cannot bypass execute permission changes");
  require((*memory)->protect({0x1000}, rwx).has_value() &&
              (*memory)->store<std::uint32_t>({0x1ffc}, 0x0c000400).has_value(),
          "place a branch at the end of a mapped page");
  state.pc = {0x1ffc};
  const let slot = (*dispatch)->run_slice(state, 2);
  require(!slot && slot.error().code == jit::dispatch_error_code::memory &&
              slot.error().pc == 0x2000 && slot.error().branch_pc == 0x1ffc,
          "missing delay-slot instruction retains both fetch PCs");
}

fn cross_page_code_validation() -> void {
  namespace mem = jit::memory;
  std::array<std::byte, 3 * mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 4);
  const let rwx =
      mem::permission::read | mem::permission::write | mem::permission::execute;
  require(
      memory && (*memory)->map_ram({0x1000}, {0}, rwx).has_value() &&
          (*memory)->map_ram({0x2000}, {2 * mem::page_size}, rwx).has_value() &&
          (*memory)->map_ram({0x3000}, {mem::page_size}, rwx).has_value() &&
          (*memory)->map_ram({0x4000}, {2 * mem::page_size}, rwx).has_value(),
      "map noncontiguous executable pages and aliases");
  require(
      (*memory)->store<std::uint32_t>({0x1ffc}, 0x34010001).has_value() &&
          (*memory)->store<std::uint32_t>({0x2000}, 0x34020002).has_value() &&
          (*memory)->store<std::uint32_t>({0x2004}, 0x70000000).has_value() &&
          (*memory)->store<std::uint32_t>({0x3000}, 0x34020004).has_value() &&
          (*memory)->store<std::uint32_t>({0x3004}, 0x70000000).has_value(),
      "install two-page guest code");
  let dispatch = jit::memory_dispatcher::create(**memory);
  require(dispatch.has_value(), "create two-page dispatcher");
  mips::state state;
  for (unsigned pass = 0; pass < 2; pass++) {
    state.pc = {0x1ffc};
    const let result = (*dispatch)->run_slice(state, 2);
    require(result && result->compiled == (pass == 0 ? 1u : 0u) &&
                state.registers[1] == 1 && state.registers[2] == 2,
            "validate and reuse code spanning noncontiguous physical pages");
  }
  require((*memory)->store<std::uint32_t>({0x4000}, 0x34020003).has_value(),
          "modify the second executable page through its alias");
  state.pc = {0x1ffc};
  const let changed = (*dispatch)->run_slice(state, 2);
  require(changed && changed->compiled == 1 && state.registers[2] == 3,
          "second-page byte changes invalidate a cached block");
  require((*memory)->protect({0x2000}, mem::permission::read).has_value(),
          "revoke second-page execution");
  state.pc = {0x1ffc};
  const let denied = (*dispatch)->run_slice(state, 2);
  require(!denied && denied.error().pc == 0x2000 &&
              denied.error().access.reason == mem::fault_reason::permission,
          "cached backing pointers do not bypass revoked execute permissions");
  require((*memory)->unmap({0x2000}).has_value() &&
              (*memory)->map_ram({0x2000}, {mem::page_size}, rwx).has_value(),
          "remap second-page physical backing");
  state.pc = {0x1ffc};
  const let remapped = (*dispatch)->run_slice(state, 2);
  require(remapped && state.registers[2] == 4,
          "mapping changes replace cached backing pointers before dereference");
}

fn handoff_boundary() -> void {
  namespace mem = jit::memory;
  std::array<std::byte, mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 1);
  require(memory &&
              (*memory)
                  ->map_ram({0x1000}, {0},
                            mem::permission::read | mem::permission::write)
                  .has_value(),
          "map handoff code");
  constexpr std::array words{mips::instruction_word{0x0c004000},
                             mips::instruction_word{0}};
  for (std::size_t i = 0; i < words.size(); i++)
    require((*memory)
                ->store<std::uint32_t>({0x1000 + i * 4}, words[i].bits)
                .has_value(),
            "load handoff code");
  require(
      (*memory)
          ->protect({0x1000}, mem::permission::read | mem::permission::execute)
          .has_value(),
      "protect handoff code");
  let program = jit::dispatcher::create(words, {0x1000}, **memory,
                                         mips::guest_address{0x10000});
  require(program.has_value(), "create configured handoff");
  mips::state state{};
  state.pc = {0x1000};
  const let short_slice = (*program)->run_slice(state, 1);
  require(short_slice &&
              short_slice->reason == jit::stop_reason::budget_exhausted &&
              state.registers[31] == 0,
          "handoff cannot bypass the call budget");
  const let result = (*program)->run_slice(state, 2);
  require(result && result->reason == jit::stop_reason::handoff &&
              result->instructions == 2 && state.pc.value == 0x10000 &&
              state.registers[31] == 0x1008,
          "handoff stops after the call and its delay slot");
  const let stopped = (*program)->run_slice(state, 0);
  require(stopped && stopped->reason == jit::stop_reason::handoff &&
              stopped->instructions == 0,
          "handoff does not execute target instructions on subsequent slices");
  require(!jit::dispatcher::create(words, {0x1000}, **memory,
                                   mips::guest_address{0x10001}) &&
              !jit::dispatcher::create(words, {0x1000}, **memory,
                                       mips::guest_address{0x1004}),
          "reject unaligned and in-image handoff targets");
  let wrong = jit::dispatcher::create(words, {0x1000}, **memory,
                                       mips::guest_address{0x20000});
  require(wrong.has_value(), "create different handoff target");
  state = {};
  state.pc = {0x1000};
  const let rejected = (*wrong)->run_slice(state, 10);
  require(!rejected &&
              rejected.error().code == jit::dispatch_error_code::invalid_pc &&
              rejected.error().pc == 0x10000,
          "an unconfigured out-of-image jump is still an error");
}

fn register_call() -> void {
  namespace mem = jit::memory;
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create JALR compiler");
  for (unsigned scenario = 0; scenario < 4; scenario++) {
    alignas(std::max_align_t) std::array<std::byte, 8192> storage;
    std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                              std::pmr::null_memory_resource()};
    const unsigned destination = scenario == 3 ? 5 : 31;
    const std::array words{
        mips::instruction_word{(8u << 21) | (destination << 11) | 9u},
        mips::instruction_word{scenario == 1 ? 0x34080007u
                               : scenario == 2
                                   ? 0xdc220000u
                                   : 0x34010000u | (destination << 21)}};
    let block = mips::translate(words, {0x1000}, arena);
    require(block.has_value(), "translate JALR");
    let native = compiler->compile(*block, arena);
    require(native.has_value(), "compile JALR");
    std::array<std::byte, mem::page_size> ram{};
    let memory = mem::address_space::create(ram, 1);
    require(memory.has_value(), "create JALR memory");
    std::array<std::uint64_t, 32> state{};
    state[8] = 0xffffffff80008000ULL;
    state[1] = 0x9000;
    state[31] = 0x1234;
    const let result = native->run(state, **memory);
    if (scenario == 2)
      require(!result && result.error().pc == 0x1004 &&
                  result.error().branch_pc == 0x1000 && state[31] == 0x1008,
              "JALR writes its link before a faulting delay slot");
    else
      require(
          result && *result == 0xffffffff80008000ULL &&
              state[destination] == 0x1008 &&
              (scenario == 1 ? state[8] == 7 : state[1] == 0x1008) &&
              (scenario != 3 || state[31] == 0x1234),
          "JALR captures its target and exposes the selected link in its slot");
  }
}

fn register_jump() -> void {
  namespace mem = jit::memory;
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create register jump compiler");
  for (unsigned scenario = 0; scenario < 2; scenario++) {
    alignas(std::max_align_t) std::array<std::byte, 8192> storage;
    std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                              std::pmr::null_memory_resource()};
    const std::array words{
        mips::instruction_word{0x03e00008},
        mips::instruction_word{scenario == 0 ? 0x341f0007u : 0xdc220000u}};
    let block = mips::translate(words, {0x1000}, arena);
    require(block && block->indirect(), "translate JR");
    require(ir::optimize(*block, ir::fold_constants).has_value(),
            "optimize JR");
    let native = compiler->compile(*block, arena);
    require(native.has_value(), "compile JR");
    std::array<std::byte, mem::page_size> ram{};
    let memory = mem::address_space::create(ram, 1);
    require(memory.has_value(), "create JR memory");
    std::array<std::uint64_t, 32> state{};
    state[31] = 0xffffffffbfc00c20ULL;
    state[1] = 0x9000;
    const let result = native->run(state, **memory);
    if (scenario == 0)
      require(result && *result == 0xffffffffbfc00c20ULL && state[31] == 7,
              "JR captures target before delay slot overwrites the register");
    else
      require(!result && result.error().pc == 0x1004 &&
                  result.error().branch_pc == 0x1000,
              "JR slot faults retain branch PC");
    const let missing =
        mips::translate(std::span{words}.first(1), {0x1000}, arena);
    require(!missing &&
                missing.error().code == mips::error_code::missing_delay_slot,
            "JR requires a delay slot");
    const std::array invalid{mips::instruction_word{0x03e10008}, words[1]};
    const let rejected = mips::translate(invalid, {0x1000}, arena);
    require(!rejected &&
                rejected.error().code == mips::error_code::invalid_encoding,
            "JR rejects reserved register fields");
  }
}

fn direct_jump() -> void {
  alignas(std::max_align_t) std::array<std::byte, 8192> storage;
  std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                            std::pmr::null_memory_resource()};
  constexpr std::array words{mips::instruction_word{0x0bf0030c},
                             mips::instruction_word{0x34010007},
                             mips::instruction_word{0xffffffff}};
  let block = mips::translate(words, {0xffffffffbfc00c08ULL}, arena);
  let compiler = amd64::compiler::create();
  require(block && compiler && block->info().instruction_count == 2,
          "J includes only its delay slot");
  let native = compiler->compile(*block, arena);
  require(native.has_value(), "compile J");
  std::array<std::uint64_t, 32> state{};
  state[31] = 0x1234;
  const let result = native->run(state);
  require(result && *result == 0xffffffffbfc00c30ULL && state[1] == 7 &&
              state[31] == 0x1234,
          "J executes its slot without writing a link");
  const let missing =
      mips::translate(std::span{words}.first(1), {0x1000}, arena);
  require(!missing &&
              missing.error().code == mips::error_code::missing_delay_slot,
          "J requires a delay slot");
}

fn jump_and_link() -> void {
  namespace mem = jit::memory;
  constexpr std::uint64_t pc = 0xffffffffbfc00000ULL;
  require(mips::jump_target({0x0ff004a8}, {pc}).value ==
                  0xffffffffbfc012a0ULL &&
              mips::jump_target({0x0c000000}, {0xffffffff8ffffffcULL}).value ==
                  0xffffffff90000000ULL,
          "JAL target preserves upper PC+4 bits");
  let compiler = amd64::compiler::create();
  require(compiler.has_value(), "create JAL compiler");
  for (unsigned scenario = 0; scenario < 3; scenario++) {
    alignas(std::max_align_t) std::array<std::byte, 8192> storage;
    std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(),
                                              std::pmr::null_memory_resource()};
    const std::array words{mips::instruction_word{0x0ff004a8},
                           mips::instruction_word{scenario == 0 ? 0x37e10000u
                                                  : scenario == 1
                                                      ? 0x341f0007u
                                                      : 0xdc620000u},
                           mips::instruction_word{0xffffffff}};
    let block = mips::translate(words, {pc}, arena);
    require(block && block->jump() && block->info().instruction_count == 2,
            "JAL ends block after delay slot");
    require(std::format("{}", *block).find("jump 0xffffffffbfc012a0") !=
                std::string::npos,
            "print unconditional IR exit");
    require(ir::optimize(*block, ir::fold_constants).has_value(),
            "optimize JAL IR");
    let native = compiler->compile(*block, arena);
    require(native.has_value(), "compile JAL");
    std::array<std::byte, mem::page_size> ram{};
    let memory = mem::address_space::create(ram, 1);
    require(memory.has_value(), "create JAL memory");
    std::array<std::uint64_t, 32> state{};
    state[3] = 0x9000;
    const let result = native->run(state, **memory);
    if (scenario == 2)
      require(!result && result.error().pc == pc + 4 &&
                  result.error().branch_pc == pc && state[31] == pc + 8,
              "faulting JAL slot preserves link and branch metadata");
    else
      require(result && *result == 0xffffffffbfc012a0ULL &&
                  state[31] == (scenario == 0 ? pc + 8 : 7) &&
                  (scenario != 0 || state[1] == pc + 8),
              "delay slot can read or overwrite JAL link");
    const let missing = mips::translate(std::span{words}.first(1), {pc}, arena);
    require(!missing &&
                missing.error().code == mips::error_code::missing_delay_slot,
            "JAL requires delay slot");
    for (const std::uint32_t nested : {0x08000000u, 0x0c000000u, 0x10000000u}) {
      const std::array invalid{words[0], mips::instruction_word{nested}};
      const let rejected = mips::translate(invalid, {pc}, arena);
      require(!rejected && rejected.error().code ==
                               mips::error_code::branch_in_delay_slot,
              "reject control transfer in JAL delay slot");
    }
  }
  std::array<std::byte, mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 1);
  constexpr let rw = mem::permission::read | mem::permission::write;
  constexpr let rx = mem::permission::read | mem::permission::execute;
  require(memory && (*memory)->map_ram({0x1000}, {0}, rw).has_value(),
          "map JAL dispatch image");
  constexpr std::array words{
      mips::instruction_word{0x0c000403}, mips::instruction_word{0x37e10000},
      mips::instruction_word{0x34020063}, mips::instruction_word{0x34030007}};
  for (std::size_t i = 0; i < words.size(); i++)
    require((*memory)
                ->store<std::uint32_t>({0x1000 + i * 4}, words[i].bits)
                .has_value(),
            "load JAL dispatch image");
  require((*memory)->protect({0x1000}, rx).has_value(), "protect JAL image");
  let dispatch = jit::dispatcher::create(words, {0x1000}, **memory);
  require(dispatch.has_value(), "create JAL dispatcher");
  mips::state state{};
  state.pc = {0x1000};
  const let short_slice = (*dispatch)->run_slice(state, 1);
  require(short_slice && short_slice->instructions == 0 &&
              state.registers[31] == 0,
          "budget cannot split JAL and slot");
  const let result = (*dispatch)->run_slice(state, 10);
  require(result && result->instructions == 3 &&
              state.registers[31] == 0x1008 && state.registers[1] == 0x1008 &&
              state.registers[2] == 0 && state.registers[3] == 7,
          "dispatcher follows JAL target after slot");
}

fn instruction_fetch_faults() -> void {
  namespace mem = jit::memory;
  constexpr let rw = mem::permission::read | mem::permission::write;
  constexpr let rx = mem::permission::read | mem::permission::execute;
  for (unsigned scenario = 0; scenario < 6; scenario++) {
    const bool branch = scenario % 3 != 0;
    const bool denied = scenario >= 3;
    std::array<std::byte, 2 * mem::page_size> ram{};
    let memory = mem::address_space::create(ram, 2);
    require(memory && (*memory)->map_ram({0x1000}, {0}, rw).has_value(),
            "map boundary fetch code");
    const std::array words{mips::instruction_word{0x34010007},
                           mips::instruction_word{scenario % 3 == 2
                                                      ? 0x0c000000u
                                                  : branch ? 0x10000000u
                                                           : 0u},
                           mips::instruction_word{0}};
    for (std::size_t i = 0; i < 2; i++)
      require((*memory)
                  ->store<std::uint32_t>({0x1ff8 + i * 4}, words[i].bits)
                  .has_value(),
              "load boundary fetch code");
    require((*memory)->protect({0x1000}, rx).has_value(),
            "protect boundary code");
    if (denied)
      require((*memory)
                  ->map_ram({0x2000}, {mem::page_size}, mem::permission::read)
                  .has_value(),
              "map non-executable next page");
    let dispatch = jit::dispatcher::create(words, {0x1ff8}, **memory);
    require(dispatch.has_value(), "create boundary dispatcher");
    mips::state state{};
    state.pc = {0x1ff8};
    const let result = (*dispatch)->run_slice(state, 100);
    require(!result &&
                result.error().code == jit::dispatch_error_code::memory &&
                result.error().access.reason ==
                    (denied ? mem::fault_reason::permission
                            : mem::fault_reason::unmapped) &&
                result.error().pc == 0x2000 && state.pc.value == 0x2000 &&
                state.registers[1] == 7,
            "later fetch fault preserves earlier instructions and fault PC");
    require(result.error().branch_pc ==
                (branch ? std::optional<std::uint64_t>{0x1ffc} : std::nullopt),
            "delay-slot fetch fault retains branch PC");
  }
}

fn queued_uart_dispatch() -> void {
  namespace mem = jit::memory;
  constexpr std::array words{
      mips::instruction_word{0x34018000}, // UART base
      mips::instruction_word{0x34030000}, // byte counter
      mips::instruction_word{0x34041388}, // 5000 bytes
      mips::instruction_word{0x34050001}, // stride
      mips::instruction_word{0x8c220004}, // poll TX ready
      mips::instruction_word{0x1040fffe}, // retry when full
      mips::instruction_word{0},
      mips::instruction_word{0xac230000}, // transmit counter low byte
      mips::instruction_word{0x0065182d}, // increment counter
      mips::instruction_word{0x1464fffa}, // next byte
      mips::instruction_word{0}};
  std::array<std::byte, mem::page_size> ram{};
  jit::uart console{{0x100000}};
  let memory = mem::address_space::create(ram, 2);
  require(memory.has_value(), "create queued dispatch memory");
  let &guest = **memory;
  constexpr let rw = mem::permission::read | mem::permission::write;
  require(guest.map_ram({0x1000}, {0}, rw).has_value() &&
              guest.map_device({0x8000}, {0x100000}, console.endpoint(), rw)
                  .has_value(),
          "map queued dispatch code and UART");
  for (std::size_t i = 0; i < words.size(); i++)
    require(
        guest.store<std::uint32_t>({0x1000 + i * 4}, words[i].bits).has_value(),
        "load queued dispatch program");
  require(
      guest.protect({0x1000}, mem::permission::read | mem::permission::execute)
          .has_value(),
      "protect queued dispatch code");
  for (std::size_t i = 0; i < jit::uart::capacity; i++)
    require(guest.store<std::uint32_t>({0x8000}, 0xaa).has_value(),
            "prefill TX to exercise guest backpressure");
  let dispatch = jit::dispatcher::create(words, {0x1000}, guest);
  require(dispatch.has_value(), "create queued UART dispatcher");
  mips::state state{};
  state.pc = {0x1000};
  const let blocked = (*dispatch)->run_slice(state, 1000, 64);
  require(blocked && blocked->reason == jit::stop_reason::yielded &&
              state.registers[3] == 0 &&
              console.pending() == jit::uart::capacity,
          "guest polling yields while TX is full without dropping bytes");
  const let discard =
      console.drain([](std::span<const std::byte> bytes) noexcept
                        -> std::expected<std::size_t, int> {
        for (const let byte : bytes)
          require(byte == std::byte{0xaa}, "prefilled output survives polling");
        return bytes.size();
      });
  require(discard && *discard == jit::uart::capacity, "drain prefilled TX");
  std::size_t received = 0;
  std::uint64_t instructions = 0;
  bool completed = false;
  for (std::size_t slice = 0; slice < 1000; slice++) {
    const let result =
        (*dispatch)->run_slice(state, 1'000'000 - instructions, 64);
    require(result.has_value(), "execute queued UART slice");
    instructions += result->instructions;
    bool acknowledged = false;
    const let drained =
        console.drain([&](std::span<const std::byte> bytes) noexcept
                          -> std::expected<std::size_t, int> {
          if (acknowledged)
            return 0;
          acknowledged = true;
          const let count = std::min<std::size_t>(7, bytes.size());
          for (std::size_t i = 0; i < count; i++) {
            require(bytes[i] == static_cast<std::byte>(received & 255),
                    "partial host drains preserve guest byte ordering");
            received++;
          }
          return count;
        });
    require(drained.has_value(), "drain queued UART slice");
    completed = result->reason == jit::stop_reason::completed;
    if (completed && console.pending() == 0)
      break;
    require(result->reason != jit::stop_reason::budget_exhausted,
            "UART polling makes progress within budget");
  }
  require(completed && received == 5000 && state.registers[3] == 5000 &&
              console.pending() == 0,
          "bounded queued execution delivers every byte across FIFO wraps");
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

fn main(int argc, char **argv) -> int {
  if (argc == 2 && std::string_view{argv[1]} == "--devices") {
    queued_uart_dispatch();
    std::println("jit: queued UART, backpressure and dispatch checks passed");
    return 0;
  }
  require(argc == 1, "unknown test arguments");
  differential();
  ownership_and_errors();
  immediate_operands();
  observed_stores();
  memory_operations();
  word_shifts();
  word_addition();
  variable_word_shifts();
  doubleword_shifts();
  register_nor();
  immediate_xor();
  cop0_writes();
  cop0_full_width();
  byte_stores();
  halfword_accesses();
  signed_byte_loads();
  mmio_console();
  example_program();
  sign_branches();
  branches_and_dispatch();
  branch_likely();
  algebraic_register_identities();
  architectural_tlb();
  prefetch_hints();
  memory_barriers();
  cp0_clock();
  cop1_transfers();
  merged_loads<false>();
  merged_loads<true>();
  merged_stores<false>();
  merged_stores<true>();
  cop1_memory_transfers();
  linked_word_operations();
  unsigned_word_load();
  indexed_cache_maintenance();
  hilo_arithmetic();
  hilo_transfers();
  checked_addition();
  exception_return();
  syscall_exceptions();
  writable_code_dispatch();
  cross_page_code_validation();
  handoff_boundary();
  register_call();
  register_jump();
  direct_jump();
  jump_and_link();
  instruction_fetch_faults();
  std::println(
      "jit: native lowering, differential and ownership checks passed");
}
