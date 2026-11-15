#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <optional>
#include <utility>
#include <print>
#include <span>
#include <string_view>
#include <dynamips/macros.h>
import jit.memory;
import jit.memory_dispatch;
import jit.dispatch;
import jit.mips64;

namespace {
namespace mem = jit::memory;
constexpr let rwx =
    mem::permission::read | mem::permission::write | mem::permission::execute;
fn require(bool valid, std::string_view what) -> void {
  if (!valid) {
    std::println(stderr, "region: {}", what);
    std::exit(1);
  }
}

struct fixture {
  std::array<std::byte, 4 * mem::page_size> ram{};
  std::unique_ptr<mem::address_space> memory;
  std::unique_ptr<jit::memory_dispatcher> dispatch;
  jit::mips64::state cpu{};

  fixture() {
    let space = mem::address_space::create(ram, 8);
    require(space.has_value(), "allocate region memory");
    memory = std::move(*space);
    for (std::size_t i = 0; i < 4; i++)
      require(memory->map_ram({0x1000 + i * 0x4000}, {i * mem::page_size}, rwx)
                  .has_value(),
              "map disjoint code pages");
    require(memory->map_ram({0xd000}, {3 * mem::page_size}, rwx).has_value(),
            "map replacement page");
    // 0x1000 root, 0x5000 nested call, 0x9000 leaf, 0xd000 spare.
    put(0x1000,
        {0x11200002, 0, 0xad070000, 0x0c001400, 0, 0x24420001, 0x08000400, 0});
    put(0x5000, {0x03e08025, 0x0c002400, 0, 0x02000008, 0});
    put(0x9000, {0x24630001, 0x03e00008, 0});
    put(0xd000, {0x24630003, 0x03e00008, 0});
    require(memory->map_ram({0x11000}, {2 * mem::page_size}, rwx).has_value(),
            "writable leaf alias");
    let created = jit::memory_dispatcher::create(*memory);
    require(created.has_value(), "create region dispatcher");
    dispatch = std::move(*created);
    cpu.pc = {0x1000};
    cpu.registers[7] = 0x24630002;
    cpu.registers[8] = 0x11000;
  }

  fn put(std::uint64_t pc, std::initializer_list<std::uint32_t> words)
      -> void {
    for (const let word : words) {
      require(memory->store<std::uint32_t>({pc}, word).has_value(),
              "install guest word");
      pc += 4;
    }
  }
  fn run(std::uint64_t slots) -> void {
    while (slots) {
      const let result = dispatch->run_slice(cpu, slots, 64);
      require(result.has_value() && result->instructions,
              "region makes bounded progress");
      slots -= result->instructions;
    }
  }
  fn train_loop(std::uint64_t slots) -> void {
    for (std::size_t i = 0; i < 4096; i++) {
      const let head = dispatch->run_slice(cpu, 1000, 2);
      require(head && head->instructions && head->instructions <= slots,
              "train the requested root before child regions can cover it");
      run(slots - head->instructions);
      require(cpu.pc.value == 0x1000, "training completes one loop");
    }
  }
  fn warm() -> void {
    train_loop(15);
    require(dispatch->region_compilations() != 0 &&
                dispatch->largest_region() >= 6,
            "warmup actually compiled the noncontiguous nested call region");
    require(cpu.pc.value == 0x1000 && cpu.registers[2] == 4096 &&
                cpu.registers[3] == 4096,
            "nested far call region preserves exact counters");
  }
};

fn internal_context_cycle() -> void {
  fixture f;
  f.put(0x1000,
        {0x24210001, 0x0c001400, 0, 0x24420001, 0x0c001400, 0, 0x08000403, 0});
  f.put(0x5000, {0x03e08021, 0x0c002400, 0, 0x0200f821, 0x03e00008, 0});
  f.put(0x9000, {0x24630001, 0x03e00008, 0});
  f.cpu = {};
  for (unsigned iteration = 0; iteration < 4096; iteration++) {
    f.cpu.pc = {0x1000};
    const let warmed = f.dispatch->run_slice(f.cpu, 256, 9);
    require(warmed && warmed->instructions == 26 && warmed->blocks == 9 &&
                f.cpu.pc.value == 0x100c,
            "train a one-time preheader followed by a nested-context cycle");
  }
  require(f.dispatch->region_block_count(0x1000) == 9 &&
              !f.dispatch->region_closed_to_root(0x1000) &&
              f.dispatch->region_backedge(0x1000) == std::size_t{4},
          "discovery retains the internal repeated-key label and both "
          "nested helper contexts");
  let state = f.cpu;
  state.pc = {0x1000};
  const let continued = f.dispatch->run_slice(state, 1000, 64);
  require(continued && continued->instructions == 180 &&
              continued->blocks == 64 && state.pc.value == 0x100c &&
              state.registers[1] == f.cpu.registers[1] + 1 &&
              state.registers[2] == f.cpu.registers[2] + 12 &&
              state.registers[3] == f.cpu.registers[3] + 13,
          "the retained internal cycle reaches the original 64-block service "
          "boundary without replaying its preheader");

  require(f.memory->map_ram({0xffffffff80000000ULL}, {3 * mem::page_size}, rwx)
              .has_value(),
          "map an actual Compare interrupt handler");
  f.put(0xffffffff80000180ULL, {0x26940001});
  let scalar = jit::memory_dispatcher::create(*f.memory, UINT16_MAX);
  require(scalar.has_value(), "create a scalar cold-block dispatcher control");
  for (const bool half : {false, true})
    for (const let count : {0u, 0xfffffffdu})
      for (const let deadline : {1u, 2u, 7u, 13u, 50u, 1000u})
        for (const std::uint64_t budget :
             {0ULL, 3ULL, 12ULL, 26ULL, 80ULL, 180ULL})
          for (const std::size_t blocks : {0u, 1u, 4u, 9u, 14u, 64u}) {
            let actual = f.cpu;
            actual.pc = {0x1000};
            actual.cop0.registers[9] = count;
            actual.cop0.registers[11] =
                static_cast<std::uint32_t>(count + deadline);
            actual.cop0.registers[12] = 0x8001;
            actual.cop0.registers[13] = 0;
            actual.cop0.half_cycle = half;
            let expected = actual;
            const let result = f.dispatch->run_slice(actual, budget, blocks);
            const let control = (*scalar)->run_slice(expected, budget, blocks);
            require(result && control &&
                        result->instructions == control->instructions &&
                        result->blocks == control->blocks &&
                        result->reason == control->reason &&
                        actual.pc.value == expected.pc.value &&
                        actual.registers == expected.registers &&
                        actual.cop0.registers == expected.cop0.registers &&
                        actual.cop0.half_cycle == expected.cop0.half_cycle,
                    "retained internal cycles match ordinary dispatcher "
                    "budgets, Compare IRQ/EPC, rollover and half phases");
            require(f.dispatch->region_backedge(0x1000) == std::size_t{4},
                    "the differential keeps its promoted internal cycle");
          }
}

fn promoted_shared_leaf() -> void {
  fixture f;
  // First train the shared leaf from a different caller. Its promoted native
  // exit and preserved first successor both point into this other context.
  f.put(0xd008, {0x24840001, 0x08003408, 0});
  f.put(0xd020, {0x40026000});
  for (std::size_t i = 0; i < 3072; i++) {
    f.cpu.pc = {0x9000};
    f.cpu.registers[31] = 0xd008;
    const let result = f.dispatch->run_slice(f.cpu, 100, 2);
    require(result && result->instructions == 6 && result->blocks == 2 &&
                f.cpu.pc.value == 0xd020,
            "warm shared leaf under a second return context");
  }
  require(f.dispatch->region_block_count(0x9000) == 2,
          "the shared child is already a native region before parent warmup");
  f.cpu = {};
  f.cpu.pc = {0x1000};
  f.cpu.registers[7] = 0x24630002;
  f.cpu.registers[8] = 0x11000;
  f.warm();
  require(f.dispatch->region_block_count(0x1000) == 6,
          "parent follows its own JAL return stack across a promoted shared "
          "leaf rather than the child's aggregate exit or other caller");
}

fn repeated_helper_contexts() -> void {
  fixture f;
  f.put(0x1000,
        {0x0c002400, 0, 0x24420001, 0x0c002400, 0, 0x24420001, 0x08000400, 0});
  f.train_loop(14);
  require(f.cpu.pc.value == 0x1000 && f.cpu.registers[2] == 8192 &&
              f.cpu.registers[3] == 8192 &&
              f.dispatch->region_block_count(0x1000) == 5 &&
              f.dispatch->region_closed_to_root(0x1000) &&
              f.dispatch->region_watch_count(0x1000) == 2,
          "two calls to the same helper retain separate return contexts and "
          "close the root loop with one watch per backing page");
  const let before = f.cpu;
  const let bounded = f.dispatch->run_slice(f.cpu, 5, 64);
  require(bounded && bounded->instructions == 5 && bounded->blocks == 2 &&
              f.cpu.pc.value == 0x1008 && f.cpu.registers[2] == 8192 &&
              f.cpu.registers[3] == 8193,
          "instruction budget stops after the first repeated helper context");
  f.cpu = before;
  const let blocks = f.dispatch->run_slice(f.cpu, 100, 3);
  require(blocks && blocks->instructions == 8 && blocks->blocks == 3 &&
              f.cpu.pc.value == 0x9000 && f.cpu.registers[2] == 8193 &&
              f.cpu.registers[3] == 8193 && f.cpu.registers[31] == 0x1014,
          "block budget stops before entering the second helper context");
  f.cpu = before;
  f.put(0x11000, {0x24630003});
  f.run(14);
  require(f.cpu.pc.value == 0x1000 && f.cpu.registers[2] == 8194 &&
              f.cpu.registers[3] == 8198,
          "alias changes refresh every duplicated helper instance");
  f.put(0xd000, {0x24630005, 0x03e00008, 0});
  require(f.memory->map_ram({0x9000}, {3 * mem::page_size}, rwx).has_value(),
          "remap a helper used under two contexts");
  f.run(14);
  require(f.cpu.pc.value == 0x1000 && f.cpu.registers[2] == 8196 &&
              f.cpu.registers[3] == 8208,
          "mapping replacement refreshes duplicated helper backing");
}

fn repeated_helper_active_write() -> void {
  fixture f;
  f.put(0x1000, {0x0c002400, 0, 0x24420001, 0xad070000, 0x0c002400, 0,
                 0x24420001, 0x08000400, 0});
  f.cpu.registers[7] = 0x24630001;
  f.train_loop(15);
  require(f.cpu.pc.value == 0x1000 && f.cpu.registers[2] == 8192 &&
              f.cpu.registers[3] == 8192 &&
              f.dispatch->region_closed_to_root(0x1000) &&
              f.dispatch->region_watch_count(0x1000) == 2,
          "warm a closed repeated-helper path with same-byte alias writes");
  f.cpu.registers[7] = 0x24630003;
  f.run(15);
  require(
      f.cpu.pc.value == 0x1000 && f.cpu.registers[2] == 8194 &&
          f.cpu.registers[3] == 8196,
      "an active alias write exits before the second stale helper instance");
}

fn repeated_helper_second_fault() -> void {
  fixture f;
  f.put(0x1000, {0x0c002400, 0, 0x24420001, 0x01605025, 0x0c002400, 0,
                 0x24420001, 0x08000400, 0});
  f.put(0x9008, {0x8d440000});
  f.cpu.registers[10] = 0x11080;
  f.cpu.registers[11] = 0x11080;
  f.train_loop(15);
  require(f.cpu.pc.value == 0x1000 && f.cpu.registers[2] == 8192 &&
              f.cpu.registers[3] == 8192 &&
              f.dispatch->region_closed_to_root(0x1000),
          "warm duplicated helper delay-slot loads");
  f.cpu.registers[11] = 0x30001;
  const let count = f.cpu.cop0.registers[9];
  const let half = f.cpu.cop0.half_cycle;
  const let fault = f.dispatch->run_slice(f.cpu, 100, 64);
  require(fault && fault->reason == jit::stop_reason::guest_exception &&
              fault->instructions == 12 && f.cpu.registers[2] == 8193 &&
              f.cpu.registers[3] == 8194 && f.cpu.registers[4] == 0 &&
              f.cpu.cop0.registers[14] == 0x9004 &&
              (f.cpu.cop0.registers[13] & 0x80000000) &&
              f.cpu.cop0.registers[9] == count + 6 &&
              f.cpu.cop0.half_cycle == half,
          "a fault in the second duplicate has precise EPC, BD and executed "
          "prefix accounting");
}

fn saved_return_guard() -> void {
  fixture f;
  f.put(0x1000,
        {0x0c002400, 0, 0x24420001, 0x0c002400, 0, 0x24420001, 0x08000400, 0});
  f.put(0x9000, {0x03e08025, 0x24630001, 0x02118021, 0x02000008, 0});
  f.train_loop(18);
  require(f.cpu.pc.value == 0x1000 && f.cpu.registers[2] == 8192 &&
              f.cpu.registers[3] == 8192 &&
              f.dispatch->region_block_count(0x1000) == 5 &&
              f.dispatch->region_closed_to_root(0x1000),
          "saved-RA JR r16 helpers close under distinct return contexts");
  f.put(0x1088, {0x70000000});
  f.cpu.registers[17] = 0x80;
  const let changed = f.dispatch->run_slice(f.cpu, 100, 64);
  require(!changed &&
              changed.error().code == jit::dispatch_error_code::translation &&
              changed.error().pc == 0x1088 && f.cpu.pc.value == 0x1088 &&
              f.cpu.registers[2] == 8192 && f.cpu.registers[3] == 8193 &&
              f.cpu.registers[16] == 0x1088,
          "JR r16 predictions still guard the actual runtime return target");
}

fn aliases_and_remapping() -> void {
  fixture f;
  f.warm();
  f.put(0x11000, {0x24630002});
  f.run(15);
  require(f.cpu.registers[2] == 4097 && f.cpu.registers[3] == 4098,
          "second virtual alias changes leaf code");
  require(f.memory->map_ram({0x9000}, {3 * mem::page_size}, rwx).has_value(),
          "remap leaf to distinct backing");
  f.run(15);
  require(f.cpu.registers[2] == 4098 && f.cpu.registers[3] == 4101,
          "all noncontiguous backing identities re-resolve after remapping");
  const let before = f.cpu.registers[2];
  f.cpu.pc = {0x9000};
  f.run(1);
  require(
      f.cpu.pc.value == 0x9004 && f.cpu.registers[2] == before &&
          f.cpu.registers[3] == 4104,
      "small budgets compile an exact straight-line prefix rather than stall");
}
fn running_write() -> void {
  fixture f;
  f.warm();
  f.cpu.registers[9] = 1;
  f.run(16);
  require(
      f.cpu.pc.value == 0x1000 && f.cpu.registers[2] == 4097 &&
          f.cpu.registers[3] == 4098,
      "store through alias exits active region before executing changed leaf");
}
fn crossing_page() -> void {
  fixture f;
  require(f.memory->map_ram({0xa000}, {3 * mem::page_size}, rwx).has_value(),
          "map leaf delay-slot second page");
  f.put(0x5004, {0x0c0027fe});
  f.put(0x9ff8, {0x24630001, 0x03e00008, 0});
  f.warm();
  f.put(0xd000, {0x24840001});
  f.run(15);
  require(f.cpu.pc.value == 0x1000 && f.cpu.registers[3] == 4097 &&
              f.cpu.registers[4] == 1,
          "alias change to a page-crossing delay slot invalidates the whole "
          "region");
}
fn speculative_mmio() -> void {
  std::uint32_t reads = 0;
  mem::device endpoint{&reads,
                       [](void *context, mem::device_access) noexcept
                           -> std::expected<std::uint64_t, mem::device_error> {
                         (*static_cast<std::uint32_t *>(context))++;
                         return 0x08000400;
                       },
                       nullptr};
  fixture f;
  require(f.memory
              ->map_device({0x15000}, {0x20000}, endpoint,
                           mem::permission::read | mem::permission::write)
              .has_value(),
          "map non-executable MMIO on an untaken branch");
  // BNE zero,zero is never taken; its target still enters the candidate queue.
  f.put(0x1000, {0x14004fff, 0, 0x0c001400, 0, 0x24420001, 0x08000400, 0});
  f.warm();
  require(reads == 0,
          "region discovery never reads an untaken non-executable device");
}
fn faulting_delay() -> void {
  fixture f;
  f.warm();
  // Replace the leaf delay slot with a faulting LW r4,0(r10).
  f.put(0x11008, {0x8d440000});
  f.cpu.registers[10] = 0x30001;
  f.cpu.registers[4] = 99;
  const let result = f.dispatch->run_slice(f.cpu, 100, 64);
  require(result && result->reason == jit::stop_reason::guest_exception &&
              f.cpu.cop0.registers[14] == 0x9004 &&
              (f.cpu.cop0.registers[13] & 0x80000000) &&
              f.cpu.registers[4] == 99 && f.cpu.registers[3] == 4097,
          "far-region delay-slot memory fault has exact EPC/BD and committed "
          "prefix");
}
fn region_translation_contract_transition() -> void {
  fixture f;

  struct source {
    fixture *owner;
    unsigned calls = 0;
    bool mutates = false;
  } callback{&f};

  const let translate =
      +[](void *opaque, mem::virtual_address address, mem::access_kind access,
          mem::access_size size) noexcept
      -> std::expected<mem::translated_page, mem::fault> {
    let &self = *static_cast<source *>(opaque);
    if ((address.value & ~std::uint64_t{mem::page_size - 1}) != 0x9000 ||
        access != mem::access_kind::execute)
      return std::unexpected(
          mem::fault{address, mem::fault_reason::unmapped, access, size});
    self.calls++;
    if (self.mutates)
      self.owner->cpu.registers[22] = 0x55;
    return mem::translated_page{
        {2 * mem::page_size + (address.value & (mem::page_size - 1))},
        self.owner->ram.data() + 2 * mem::page_size,
        nullptr};
  };
  require(f.memory->unmap({0x9000}).has_value(),
          "supply one future helper source through the translator");
  f.memory->set_translation(&callback, translate,
                            mem::translation_effect::guest_state_preserving);
  f.warm();
  require(f.dispatch->region_block_count(0x1000) >= 6 && callback.calls != 0,
          "a preserving callback may supply a cached noncontiguous region");
  const let calls = callback.calls;
  callback.mutates = true;
  f.cpu.registers[22] = 0x11;
  f.memory->set_translation(&callback, translate);
  const let actual = f.dispatch->run_slice(f.cpu, 2, 1);
  require(actual && actual->compiled == 1 && actual->instructions == 2 &&
              actual->blocks == 1 && f.cpu.pc.value == 0x100c &&
              f.cpu.registers[22] == 0x11 && callback.calls == calls &&
              f.dispatch->region_block_count(0x1000) == 0,
          "an opaque contract retires a cached region before touching any "
          "future fragment or backing stamp");
  callback.mutates = false;
  f.memory->set_translation(&callback, translate,
                            mem::translation_effect::guest_state_preserving);
  f.cpu.pc = {0x1000};
  f.train_loop(15);
  require(f.dispatch->region_block_count(0x1000) >= 6 &&
              f.cpu.registers[22] == 0x11,
          "restoring a preserving contract permits ordinary promotion again");
}

fn cpu_source_alias(bool cached_disjoint) -> void {
  struct padded_cpu {
    jit::mips64::state state{};
    std::array<std::byte, mem::page_size> padding{};
  } source;

  jit::mips64::state other{};
  std::array<std::byte, mem::page_size> ram{};
  let made = mem::address_space::create(ram, 1);
  require(made.has_value(), "allocate CPU-source fixture");
  let memory = std::move(*made);
  require(memory->map_ram({0x1000}, {0}, rwx).has_value(), "map root RAM");
  memory->set_translation(
      &source.state,
      [](void *opaque, mem::virtual_address address, mem::access_kind access,
         mem::access_size size) noexcept
          -> std::expected<mem::translated_page, mem::fault> {
        if ((address.value & ~std::uint64_t{mem::page_size - 1}) != 0x9000)
          return std::unexpected(
              mem::fault{address, mem::fault_reason::unmapped, access, size});
        let &cpu = *static_cast<jit::mips64::state *>(opaque);
        return mem::translated_page{
            {0x8000000 + (address.value & (mem::page_size - 1))},
            reinterpret_cast<std::byte *>(cpu.registers.data()),
            nullptr};
      },
      mem::translation_effect::guest_state_preserving);
  const std::array root{0x01404025u, 0x08002410u, 0u};
  for (std::size_t i = 0; i < root.size(); i++)
    require(memory->store<std::uint32_t>({0x1000 + i * 4}, root[i]).has_value(),
            "install OR r8,r10 followed by CPU-backed target");
  const let encoded = [](unsigned value) {
    const std::array bytes{std::byte{0x34}, std::byte{1},
                           std::byte{0},    static_cast<std::byte>(value),
                           std::byte{8},    std::byte{0},
                           std::byte{4},    std::byte{0}};
    std::uint64_t result;
    std::memcpy(&result, bytes.data(), sizeof(result));
    return result;
  };
  source.state.registers[8] = encoded(1);
  source.state.registers[10] = encoded(1);
  other.registers[10] = encoded(1);
  let created = jit::memory_dispatcher::create(*memory);
  require(created.has_value(), "create CPU-source dispatcher");
  let dispatch = std::move(*created);
  let &warm_state = cached_disjoint ? other : source.state;
  warm_state.pc = {0x1000};
  for (unsigned i = 0; i < 4096; i++) {
    const let warm = dispatch->run_slice(warm_state, 1000, 2);
    require(warm && warm->instructions == 6 && warm_state.pc.value == 0x1000,
            "warm source backed by a padded CPU object");
  }
  require((dispatch->region_block_count(0x1000) != 0) == cached_disjoint,
          "only a source disjoint from the active CPU may form a region");
  source.state.pc = {0x1000};
  source.state.registers[1] = 0;
  source.state.registers[10] = encoded(7);
  const let changed = dispatch->run_slice(source.state, 6, 2);
  require(changed && changed->instructions == 6 &&
              source.state.registers[1] == 7 &&
              dispatch->region_block_count(0x1000) == 0,
          "direct CPU writes use freshly validated ordinary target code");
}

fn skewed_source_versions() -> void {
  std::array<std::byte, 4 * mem::page_size> ram{};
  let made = mem::address_space::create(ram, 3, mem::backing_policy::tracked);
  require(made.has_value(), "allocate skewed executable backing");
  let memory = std::move(*made);
  for (const let [address, physical] :
       {std::pair{0x1000u, std::size_t{0}},
        std::pair{0xa000u, 3u * mem::page_size},
        std::pair{0x11000u, 2u * mem::page_size}})
    require(memory->map_ram({address}, {physical}, rwx).has_value(),
            "map root, delay page and physical code-write alias");
  memory->set_translation(
      &ram,
      [](void *opaque, mem::virtual_address address, mem::access_kind access,
         mem::access_size size) noexcept
          -> std::expected<mem::translated_page, mem::fault> {
        if ((address.value & ~std::uint64_t{mem::page_size - 1}) != 0x9000)
          return std::unexpected(
              mem::fault{address, mem::fault_reason::unmapped, access, size});
        let &bytes = *static_cast<decltype(ram) *>(opaque);
        return mem::translated_page{
            {mem::page_size + 4 + (address.value & (mem::page_size - 1))},
            bytes.data() + mem::page_size + 4,
            nullptr};
      },
      mem::translation_effect::guest_state_preserving);
  for (const let [address, word] :
       {std::pair{0x1000u, 0x08002400u}, std::pair{0x1004u, 0u},
        std::pair{0x9000u, 0x080027ffu}, std::pair{0x9004u, 0u},
        std::pair{0x9ffcu, 0x34010001u}, std::pair{0xa000u, 0x08000400u},
        std::pair{0xa004u, 0u}})
    require(memory->store<std::uint32_t>({address}, word).has_value(),
            "install separated fragments of one skewed virtual page");
  let created = jit::memory_dispatcher::create(*memory);
  require(created.has_value(), "create skewed-source dispatcher");
  let dispatch = std::move(*created);
  jit::mips64::state cpu{};
  cpu.pc = {0x1000};
  for (unsigned i = 0; i < 4096; i++) {
    const let warm = dispatch->run_slice(cpu, 1000, 3);
    require(warm && warm->instructions == 7 && cpu.pc.value == 0x1000,
            "warm a region with two physical version pages per virtual page");
  }
  require(dispatch->region_block_count(0x1000) == 3,
          "the skewed fixture exercises promoted region validation");
  require(memory->store<std::uint32_t>({0x11000}, 0x34010007).has_value(),
          "modify the second physical source page through its aligned alias");
  cpu.registers[1] = 0;
  const let changed = dispatch->run_slice(cpu, 7, 3);
  require(changed && changed->instructions == 7 && cpu.registers[1] == 7,
          "every physical page containing any region fragment is versioned");
}
} // namespace

fn main() -> int {
  cpu_source_alias(false);
  cpu_source_alias(true);
  skewed_source_versions();
  region_translation_contract_transition();
  internal_context_cycle();
  repeated_helper_contexts();
  repeated_helper_active_write();
  repeated_helper_second_fault();
  saved_return_guard();
  promoted_shared_leaf();
  aliases_and_remapping();
  running_write();
  crossing_page();
  speculative_mmio();
  faulting_delay();
  std::println("regions: nested far calls, aliases, active writes, remapping, "
               "precise delay fault and budgets passed");
}
