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
import jit.uart;
import jit.example;
import jit.dispatch;

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
      switch (next(seed) % 5) {
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
  constexpr std::array words{mips::instruction_word{0x80220001}};
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
    require(result && state[2] == static_cast<std::uint64_t>(
                                      value < 128 ? std::int64_t{value}
                                                  : std::int64_t{value} - 256),
            "LB sign extends every byte at an unaligned address");
  }
  state[1] = 0x8fff;
  const let fault = native->run(state, **memory);
  require(!fault && fault.error().pc == 0x1000 &&
              fault.error().access.address.value == 0x9000,
          "byte load reports exact unmapped address");
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
  signed_byte_loads();
  mmio_console();
  example_program();
  branches_and_dispatch();
  direct_jump();
  jump_and_link();
  instruction_fetch_faults();
  std::println(
      "jit: native lowering, differential and ownership checks passed");
}
