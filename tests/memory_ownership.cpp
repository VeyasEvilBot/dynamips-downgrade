#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <initializer_list>
#include <memory>
#include <memory_resource>
#include <optional>
#include <print>
#include <span>
#include <string_view>
#include <utility>

#include <dynamips/macros.h>

import jit.amd64;
import jit.ir;
import jit.dispatch;
import jit.memory;
import jit.memory_dispatch;
import jit.mips64;

namespace backing_proofs {

namespace mem = jit::memory;
constexpr let rwx =
    mem::permission::read | mem::permission::write | mem::permission::execute;
fn require(bool valid, std::string_view what) -> void {
  if (!valid) {
    std::println(stderr, "tracked backing: {}", what);
    std::exit(1);
  }
}

fn raw_word(std::byte *address, std::uint32_t bits) noexcept -> void {
  if constexpr (std::endian::native == std::endian::little)
    bits = std::byteswap(bits);
  std::memcpy(address, &bits, sizeof(bits));
}

struct opaque_write {
  std::byte *address = nullptr;
  std::uint32_t bits = 0;
  unsigned reads = 0;

  static fn read(void *context, mem::device_access) noexcept
      -> std::expected<std::uint64_t, mem::device_error> {
    let &self = *static_cast<opaque_write *>(context);
    self.reads++;
    if (self.address)
      raw_word(self.address, self.bits);
    return 0;
  }
};

struct basic {
  std::array<std::byte, 2 * mem::page_size> ram{};
  opaque_write callback;
  mem::device endpoint{&callback, &opaque_write::read, nullptr};
  std::unique_ptr<mem::address_space> memory;
  std::unique_ptr<jit::memory_dispatcher> dispatch;
  jit::mips64::state cpu{};

  explicit basic(std::optional<mem::backing_policy> policy = {}) {
    let space = policy ? mem::address_space::create(ram, 8, *policy)
                        : mem::address_space::create(ram, 8);
    require(space.has_value(), "allocate backing policy memory");
    memory = std::move(*space);
    require(memory->map_ram({0x1000}, {0}, rwx).has_value() &&
                memory->map_ram({0x5000}, {0}, rwx).has_value() &&
                memory->map_ram({0x9000}, {mem::page_size}, rwx).has_value() &&
                memory
                    ->map_device({0xd000}, {0x20000}, endpoint,
                                 mem::permission::read)
                    .has_value(),
            "map code aliases, replacement RAM and opaque MMIO");
    put(0x1000, {0x34030001, 0x08000400, 0});
    put(0x9000, {0x34030005, 0x08000400, 0});
    let created = jit::memory_dispatcher::create(*memory);
    require(created.has_value(), "create policy dispatcher");
    dispatch = std::move(*created);
    cpu.pc = {0x1000};
  }

  fn put(std::uint64_t pc, std::initializer_list<std::uint32_t> words)
      -> void {
    for (const let word : words) {
      require(memory->store<std::uint32_t>({pc}, word).has_value(),
              "install policy test code");
      pc += 4;
    }
  }
  fn run(std::uint64_t slots) -> void {
    const let result = dispatch->run_slice(cpu, slots, 64);
    require(result && result->instructions == slots,
            "execute exact policy test budget");
  }
};

fn borrowed_default() -> void {
  basic f;
  require(f.memory->policy() == mem::backing_policy::borrowed &&
              f.memory->requires_slice_validation(),
          "two-argument creation preserves borrowed validation by default");
  f.run(3);
  const let stamp = f.memory->stamp(std::span{f.ram}.first(4));
  const let before = stamp.value();
  raw_word(f.ram.data(), 0x34030009);
  require(stamp.value() == before, "direct host edit bypasses version counter");
  f.run(3);
  require(f.cpu.registers[3] == 9,
          "default borrowed code still detects direct edits between slices");
}

fn tracked_writers() -> void {
  basic f{mem::backing_policy::tracked};
  require(f.memory->policy() == mem::backing_policy::tracked &&
              !f.memory->requires_slice_validation(),
          "tracked validation requires an explicit fixed owner policy");
  f.put(0x1100,
        {0x34015000, 0x3c023403, 0x34420009, 0xac220000, 0x08000400, 0});
  f.run(3);
  f.cpu.pc = {0x1100};
  f.run(9);
  require(f.cpu.registers[3] == 9,
          "native CPU store through a physical alias invalidates tracked code");
  f.put(0x5000, {0x3403000b});
  f.run(3);
  require(f.cpu.registers[3] == 11,
          "host API alias stores update tracked code");
  std::array<std::byte, 4> replacement;
  raw_word(replacement.data(), 0x3403000d);
  require(f.memory->load_linked<std::uint32_t>({0x5000}).has_value() &&
              f.memory->write_physical({0}, replacement).has_value() &&
              f.memory->store_conditional<std::uint32_t>({0x5000}, 0) == false,
          "physical DMA write uses coherency and clears the reservation");
  f.run(3);
  require(f.cpu.registers[3] == 13,
          "physical DMA updates executable code across tracked slices");
}

fn opaque_effects() -> void {
  basic f{mem::backing_policy::tracked};
  f.run(3);
  f.callback.address = f.ram.data();
  f.callback.bits = 0x34030009;
  require(f.memory->load<std::uint32_t>({0xd000}) == 0,
          "invoke opaque callback between execution slices");
  f.run(3);
  require(f.callback.reads == 1 && f.cpu.registers[3] == 9,
          "opaque external epoch still forces exact tracked byte proof");

  basic active{mem::backing_policy::tracked};
  active.put(0x1000, {0x8d040000, 0x34030007, 0x08000400, 0});
  active.cpu.registers[8] = 0xd000;
  active.run(4);
  require(active.cpu.registers[3] == 7, "warm MMIO-bearing code");
  active.callback.address = active.ram.data() + 4;
  active.callback.bits = 0x3403000b;
  active.run(4);
  require(active.cpu.registers[3] == 11,
          "active opaque callback exits before executing modified successor");
}

fn changed_mappings() -> void {
  basic remap{mem::backing_policy::tracked};
  remap.run(3);
  require(remap.memory->map_ram({0x1000}, {mem::page_size}, rwx).has_value(),
          "replace tracked executable backing");
  remap.run(3);
  require(remap.cpu.registers[3] == 5,
          "mapping epoch re-resolves pointers before tracked reuse");
  require(
      remap.memory
          ->protect({0x1000}, mem::permission::read | mem::permission::write)
          .has_value(),
      "revoke tracked executable permissions");
  const let denied = remap.dispatch->run_slice(remap.cpu, 1, 64);
  require(!denied && denied.error().code == jit::dispatch_error_code::memory &&
              denied.error().access.reason == mem::fault_reason::permission,
          "tracked code cannot bypass mapping permission revocation");
}

fn rom_policy() -> void {
  for (const let policy :
       {mem::backing_policy::borrowed, mem::backing_policy::tracked}) {
    std::array<std::byte, mem::page_size> first{}, second{};
    basic f{policy};
    for (let *backing : {first.data(), second.data()}) {
      raw_word(backing, backing == first.data() ? 0x34030007u : 0x34030009u);
      raw_word(backing + 4, 0x08004400);
    }
    require(f.memory->map_rom({0x11000}, {0x10000}, first).has_value(),
            "map immutable executable ROM");
    f.cpu.pc = {0x11000};
    f.run(3);
    require(f.cpu.registers[3] == 7, "execute ROM under either policy");
    if (policy == mem::backing_policy::borrowed) {
      raw_word(first.data(), 0x3403000b);
      f.run(3);
      require(f.cpu.registers[3] == 11,
              "borrowed ROM still observes direct between-slice edits");
    }
    require(f.memory->map_rom({0x11000}, {0x10000}, second).has_value(),
            "replace ROM backing through mapping API");
    f.run(3);
    require(f.cpu.registers[3] == 9,
            "ROM remapping is observed even with no RAM page stamps");
  }
}

} // namespace backing_proofs

namespace watch_proofs {

namespace mem = jit::memory;
constexpr let rw = mem::permission::read | mem::permission::write;
constexpr std::uint64_t base = 0x100000, alias = 0x200000;
fn require(bool condition, std::string_view message) -> void {
  if (!condition) {
    std::println(stderr, "state watch: {}", message);
    std::exit(1);
  }
}

struct alignas(mem::page_size) storage {
  jit::mips64::state cpu{};
  std::array<std::byte, mem::page_size> extra{};
};

struct fixture {
  storage backing;
  std::span<std::byte> bytes = std::as_writable_bytes(std::span{&backing, 1});
  std::span<const std::byte> registers =
      std::as_bytes(std::span{backing.cpu.registers});
  std::unique_ptr<mem::address_space> memory;

  fixture() {
    const let pages = bytes.size() / mem::page_size;
    require(bytes.size() % mem::page_size == 0, "complete backing pages");
    let space = mem::address_space::create(bytes, 2 * pages);
    require(space.has_value(), "allocate state-alias memory");
    memory = std::move(*space);
    for (std::size_t i = 0; i < pages; i++)
      require(
          memory->map_ram({base + i * mem::page_size}, {i * mem::page_size}, rw)
                  .has_value() &&
              memory
                  ->map_ram({alias + i * mem::page_size}, {i * mem::page_size},
                            rw)
                  .has_value(),
          "map guest RAM and a physical alias");
  }

  fn offset(const std::byte *pointer) const -> std::uint64_t {
    return reinterpret_cast<std::uintptr_t>(pointer) -
           reinterpret_cast<std::uintptr_t>(bytes.data());
  }
  fn address(const std::byte *pointer, bool second = false) const
      -> mem::virtual_address {
    return {(second ? alias : base) + offset(pointer)};
  }

  static fn before_external(void *context) noexcept -> void {
    (*static_cast<unsigned *>(context))++;
  }
};

fn independent_ranges() -> void {
  fixture f;
  const std::array code{
      std::span<const std::byte>{f.backing.extra}.subspan(64, 16)};
  unsigned callbacks = 0;
  mem::write_monitor monitor{code, false, &callbacks,
                             &fixture::before_external};
  require(monitor.architectural_state.empty(),
          "old aggregate initialization defaults to no state observation");
  monitor.architectural_state = f.registers;
  const let stamp = f.memory->stamp(f.bytes);
  const let epoch = f.memory->external_generation();
  const let before = stamp.value();
  {
    const let scope = f.memory->monitor_writes(&monitor);
    require(f.memory->store<std::uint32_t>(
                        f.address(f.backing.extra.data() + 128), 7)
                    .has_value() &&
                !monitor.modified && stamp.value() == before + 1,
            "ordinary RAM outside code and state leaves the observer clean");
    require(
        f.memory->store<std::uint64_t>(f.address(f.registers.data() + 16, true),
                                       0x123456789abcdef0ULL)
                .has_value() &&
            monitor.modified && stamp.value() == before + 2 && callbacks == 0 &&
            f.memory->external_generation() == epoch,
        "RAM alias into CPU state is observed independently of code ranges");
    require(
        f.memory->store<std::uint32_t>(f.address(f.backing.extra.data() + 128),
                                       8)
                .has_value() &&
            stamp.value() == before + 3,
        "writes still advance versions after an observer is already modified");
  }
}

fn boundaries() -> void {
  fixture f;
  const let watched = f.registers.subspan(16, 16);
  mem::write_monitor monitor{};
  monitor.architectural_state = watched;
  const let scope = f.memory->monitor_writes(&monitor);
  const let write = [&](const std::byte *pointer, bool overlap) {
    monitor.modified = false;
    require(
        f.memory->store<std::uint8_t>(f.address(pointer), 0x7f).has_value() &&
            monitor.modified == overlap,
        "half-open state span boundaries match exact byte overlap");
  };
  write(watched.data() - 1, false);
  write(watched.data(), true);
  write(watched.data() + watched.size() - 1, true);
  write(watched.data() + watched.size(), false);
  monitor.modified = false;
  const std::array payload{std::byte{1}, std::byte{2}, std::byte{3},
                           std::byte{4}};
  require(f.memory->store_bytes(f.address(watched.data() - 2), payload)
                  .has_value() &&
              monitor.modified,
          "partial state overlap at the beginning is observed");
  monitor.modified = false;
  require(f.memory->store_bytes(f.address(watched.data() + watched.size() - 2),
                                payload)
                  .has_value() &&
              monitor.modified,
          "partial state overlap at the end is observed");
  monitor.architectural_state = watched.first(0);
  monitor.modified = false;
  require(
      f.memory->store<std::uint8_t>(f.address(watched.data()), 9).has_value() &&
          !monitor.modified,
      "an empty state span does not observe its data address");
  monitor.architectural_state = watched;
  const let before = f.memory->stamp(f.bytes).value();
  require(
      f.memory->write_physical({f.offset(watched.data())}, {}).has_value() &&
          !monitor.modified && f.memory->stamp(f.bytes).value() == before,
      "empty writes preserve observation and versions");
}

fn nested_scopes() -> void {
  fixture f;
  const std::array outer_code{
      std::span<const std::byte>{f.backing.extra}.subspan(32, 16)};
  const std::array inner_code{
      std::span<const std::byte>{f.backing.extra}.subspan(192, 16)};
  mem::write_monitor outer{outer_code}, inner{inner_code};
  outer.architectural_state = f.registers;
  inner.architectural_state = f.registers.subspan(32, 8);
  const let outer_scope = f.memory->monitor_writes(&outer);
  {
    const let inner_scope = f.memory->monitor_writes(&inner);
    require(
        f.memory->store<std::uint32_t>(f.address(f.registers.data() + 16), 1)
                .has_value() &&
            !outer.modified && !inner.modified,
        "only the innermost monitor observes state writes");
    require(
        f.memory->store<std::uint32_t>(f.address(f.registers.data() + 32), 2)
                .has_value() &&
            !outer.modified && inner.modified,
        "nested state observer has its own borrowed span");
  }
  require(f.memory->store<std::uint32_t>(f.address(f.registers.data() + 16), 3)
                  .has_value() &&
              outer.modified,
          "unwinding restores the previous state observer");
  outer.modified = false;
  {
    const let disabled = f.memory->monitor_writes(nullptr);
    require(
        f.memory->store<std::uint32_t>(f.address(f.registers.data() + 16), 4)
                .has_value() &&
            !outer.modified,
        "an explicit null scope disables an outer observer");
  }
  require(f.memory->store<std::uint32_t>(f.address(outer_code[0].data()), 5)
                  .has_value() &&
              outer.modified,
          "unwinding also restores the original code ranges");
}

fn linked_stores() -> void {
  fixture f;
  mem::write_monitor monitor{};
  monitor.architectural_state = f.registers;
  const let scope = f.memory->monitor_writes(&monitor);
  const let target = f.registers.data() + 16;
  const let stamp = f.memory->stamp(f.registers);
  const let before = stamp.value();
  require(
      f.memory->store_conditional<std::uint32_t>(f.address(target), 1) ==
              false &&
          !monitor.modified && stamp.value() == before,
      "failed conditional stores do not observe a write or increment versions");
  require(
      f.memory->load_linked<std::uint32_t>(f.address(target)).has_value() &&
          f.memory->store_conditional<std::uint32_t>(f.address(target, true),
                                                     2) == true &&
          monitor.modified && stamp.value() == before + 1,
      "successful conditional alias stores observe state and update versions");
  monitor.modified = false;
  require(
      f.memory->load_linked<std::uint32_t>(f.address(target)).has_value() &&
          f.memory->store<std::uint32_t>(f.address(target, true), 3)
              .has_value() &&
          f.memory->store_conditional<std::uint32_t>(f.address(target), 4) ==
              false &&
          monitor.modified && stamp.value() == before + 2,
      "ordinary state alias stores preserve reservation clearing semantics");
}

fn mutable_code_descriptors() -> void {
  fixture f;
  std::array code{std::span<const std::byte>{f.backing.extra}.subspan(32, 16)};
  const std::array replacement{
      std::span<const std::byte>{f.backing.extra}.subspan(256, 16)};
  mem::write_monitor monitor{code};
  const let scope = f.memory->monitor_writes(&monitor);
  code[0] = std::span<const std::byte>{f.backing.extra}.subspan(192, 16);
  require(f.memory->store<std::uint32_t>(f.address(code[0].data()), 1)
                  .has_value() &&
              monitor.modified,
          "changing borrowed code descriptors inside a scope remains visible");
  monitor.modified = false;
  require(
      f.memory->store<std::uint32_t>(f.address(f.backing.extra.data() + 32), 2)
              .has_value() &&
          !monitor.modified,
      "replaced descriptors no longer watch the old bytes");
  monitor.ranges = replacement;
  require(f.memory->store<std::uint32_t>(f.address(replacement[0].data()), 3)
                  .has_value() &&
              monitor.modified,
          "rebinding a live code-range span preserves the existing contract");
}

struct translator {
  fixture &owner;
  mem::write_monitor *monitor;
  unsigned *hooks;
  unsigned calls = 0;
  bool observed_before_mutation = true;
  bool fail = false;

  static fn translate(void *context, mem::virtual_address address,
                       mem::access_kind access, mem::access_size size) noexcept
      -> std::expected<mem::translated_page, mem::fault> {
    let &self = *static_cast<translator *>(context);
    self.calls++;
    self.observed_before_mutation = self.observed_before_mutation &&
                                    self.monitor->modified &&
                                    *self.hooks == self.calls;
    self.owner.backing.cpu.registers[5] = 0x12345678;
    if (self.fail)
      return std::unexpected(
          mem::fault{address, mem::fault_reason::tlb_miss, access, size});
    return mem::translated_page{{address.value & (mem::page_size - 1)},
                                self.owner.bytes.data(),
                                nullptr};
  }
};

fn translation_effects() -> void {
  for (const bool fail : {false, true}) {
    fixture f;
    unsigned hooks = 0;
    mem::write_monitor monitor{{}, false, &hooks, &fixture::before_external};
    monitor.architectural_state = f.registers;
    translator callback{f, &monitor, &hooks};
    callback.fail = fail;
    f.memory->set_translation(&callback, &translator::translate);
    const let before = f.memory->external_generation();
    {
      const let scope = f.memory->monitor_writes(&monitor);
      const let loaded = f.memory->load<std::uint32_t>({0x900000});
      const let fetched = f.memory->fetch<std::uint32_t>({0x900000});
      const let stored = f.memory->store<std::uint32_t>({0x900000}, 7);
      require(loaded.has_value() == !fail && fetched.has_value() == !fail &&
                  stored.has_value() == !fail && monitor.modified &&
                  callback.calls == 3 && callback.observed_before_mutation &&
                  hooks == 3 && f.memory->external_generation() == before + 3 &&
                  f.backing.cpu.registers[5] == 0x12345678,
              "arbitrary translation effects are visible before callback CPU "
              "mutation");
      if (fail)
        require(
            loaded.error().reason == mem::fault_reason::tlb_miss &&
                loaded.error().address.value == 0x900000 &&
                loaded.error().access == mem::access_kind::read &&
                loaded.error().size == mem::access_size::word &&
                fetched.error().access == mem::access_kind::execute &&
                stored.error().access == mem::access_kind::write,
            "translation barriers preserve typed faults and access contexts");
    }
    f.memory->set_translation(nullptr, nullptr);
  }
}

} // namespace watch_proofs

namespace translation_proofs {

namespace mem = jit::memory;
constexpr std::uint64_t address = 0x600000;
fn require(bool condition, std::string_view message) -> void {
  if (!condition) {
    std::println(stderr, "translation effect: {}", message);
    std::exit(1);
  }
}

struct clock_hook {
  jit::mips64::state &cpu;
  unsigned calls = 0;

  static fn account(void *context) noexcept -> void {
    let &self = *static_cast<clock_hook *>(context);
    self.calls++;
    self.cpu.cop0.advance_cycles(7);
  }
};

struct device_state {
  clock_hook &hook;
  mem::write_monitor &monitor;
  std::uint64_t value = 0x1234;
  unsigned reads = 0, writes = 0;
  bool observed_before_effect = true;
  fn observe() noexcept -> void {
    observed_before_effect = observed_before_effect && monitor.modified &&
                             hook.calls == reads + writes;
    hook.cpu.registers[5]++;
  }

  static fn read(void *context, mem::device_access) noexcept
      -> std::expected<std::uint64_t, mem::device_error> {
    let &self = *static_cast<device_state *>(context);
    self.reads++;
    self.observe();
    return self.value;
  }

  static fn write(void *context, mem::device_access,
                   std::uint64_t value) noexcept
      -> std::expected<void, mem::device_error> {
    let &self = *static_cast<device_state *>(context);
    self.writes++;
    self.observe();
    self.value = value;
    return {};
  }
};

struct translation {
  std::byte *host;
  std::uint64_t physical = 0;
  mem::device *device = nullptr;
  unsigned calls = 0;
  bool fail = false;

  static fn resolve(void *context, mem::virtual_address where,
                     mem::access_kind access, mem::access_size size) noexcept
      -> std::expected<mem::translated_page, mem::fault> {
    let &self = *static_cast<translation *>(context);
    self.calls++;
    if (self.fail)
      return std::unexpected(
          mem::fault{where, mem::fault_reason::tlb_invalid, access, size});
    return mem::translated_page{
        {self.physical + (where.value & (mem::page_size - 1))},
        self.host,
        self.device};
  }
};

struct fixture {
  std::array<std::byte, 2 * mem::page_size> ram{};
  jit::mips64::state cpu{};
  mem::write_monitor monitor{};
  clock_hook hook{cpu};
  device_state device{hook, monitor};
  mem::device endpoint{&device, &device_state::read, &device_state::write};
  translation original{ram.data()};
  translation replacement{ram.data() + mem::page_size, mem::page_size};
  std::unique_ptr<mem::address_space> memory;

  explicit fixture(mem::backing_policy policy = mem::backing_policy::borrowed) {
    let space = mem::address_space::create(ram, 0, policy);
    require(space.has_value(), "create translation effect memory");
    memory = std::move(*space);
    monitor.context = &hook;
    monitor.before_external = &clock_hook::account;
    monitor.architectural_state = std::as_bytes(std::span{cpu.registers});
    cpu.cop0.registers[9] = 10;
    cpu.cop0.registers[11] = 1000;
  }

  fn put(std::uint64_t physical, std::uint32_t bits) -> void {
    if constexpr (std::endian::native == std::endian::little)
      bits = std::byteswap(bits);
    require(
        memory->write_physical({physical}, std::as_bytes(std::span{&bits, 1}))
            .has_value(),
        "write physical fixture bytes through the memory API");
  }
};

fn ram_and_faults() -> void {
  for (const let policy :
       {mem::backing_policy::borrowed, mem::backing_policy::tracked})
    for (const bool pure : {false, true})
      for (const bool fail : {false, true}) {
        fixture f{policy};
        f.original.fail = fail;
        if (pure)
          f.memory->set_translation(
              &f.original, &translation::resolve,
              mem::translation_effect::guest_state_preserving);
        else
          f.memory->set_translation(&f.original, &translation::resolve);
        const let before = f.memory->external_generation();
        {
          const let scope = f.memory->monitor_writes(&f.monitor);
          const let loaded = f.memory->load<std::uint32_t>({address});
          const let fetched = f.memory->fetch<std::uint32_t>({address});
          const let stored = f.memory->store<std::uint32_t>({address}, 7);
          require(loaded.has_value() == !fail && fetched.has_value() == !fail &&
                      stored.has_value() == !fail && f.original.calls == 3,
                  "effect kind preserves valid mappings and original failures");
          require(f.monitor.modified == !pure &&
                      f.hook.calls == (pure ? 0u : 3u) &&
                      f.memory->external_generation() ==
                          before + (pure ? 0u : 3u) &&
                      f.cpu.cop0.registers[9] == (pure ? 10u : 20u) &&
                      f.cpu.cop0.half_cycle == !pure,
                  "pure RAM and fault callbacks leave observation and clock "
                  "untouched");
          if (fail)
            require(
                loaded.error().reason == mem::fault_reason::tlb_invalid &&
                    loaded.error().address.value == address &&
                    loaded.error().access == mem::access_kind::read &&
                    loaded.error().size == mem::access_size::word &&
                    fetched.error().reason == mem::fault_reason::tlb_invalid &&
                    fetched.error().access == mem::access_kind::execute &&
                    stored.error().reason == mem::fault_reason::tlb_invalid &&
                    stored.error().access == mem::access_kind::write,
                "pure callback faults retain address, access and size");
        }
      }
}

fn mmio_effects() -> void {
  fixture f{mem::backing_policy::tracked};
  f.original.host = nullptr;
  f.original.device = &f.endpoint;
  f.original.physical = 0x20000;
  f.memory->set_translation(&f.original, &translation::resolve,
                            mem::translation_effect::guest_state_preserving);
  const let before = f.memory->external_generation();
  const let scope = f.memory->monitor_writes(&f.monitor);
  require(f.memory->load<std::uint32_t>({address}) == 0x1234 &&
              f.memory->store<std::uint32_t>({address}, 7).has_value() &&
              f.original.calls == 2 && f.device.reads == 1 &&
              f.device.writes == 1 && f.device.value == 7 &&
              f.monitor.modified && f.hook.calls == 2 &&
              f.device.observed_before_effect && f.cpu.registers[5] == 2 &&
              f.memory->external_generation() == before + 2,
          "pure translation never suppresses the actual MMIO effect barrier");
}

fn setter_invalidates() -> void {
  fixture f{mem::backing_policy::tracked};
  f.put(0, 0x11223344);
  f.put(mem::page_size, 0x55667788);
  f.memory->set_translation(&f.original, &translation::resolve,
                            mem::translation_effect::guest_state_preserving);
  require(f.memory->load<std::uint32_t>({address}) == 0x11223344 &&
              f.memory->load_linked<std::uint32_t>({address}) == 0x11223344,
          "warm RAM translation and a reservation");
  const let generation = f.memory->mapping_generation();
  const let epoch = f.memory->external_generation();
  f.memory->set_translation(&f.original, &translation::resolve,
                            mem::translation_effect::guest_state_preserving);
  require(
      f.memory->mapping_generation() != generation &&
          f.memory->external_generation() != epoch &&
          f.memory->store_conditional<std::uint32_t>({address}, 0) == false,
      "even identical preserving setters invalidate proof and reservations");
  f.memory->set_translation(&f.replacement, &translation::resolve,
                            mem::translation_effect::guest_state_preserving);
  require(f.memory->load<std::uint32_t>({address}) == 0x55667788 &&
              f.replacement.calls == 1,
          "setter changes cannot reuse a stale cached host page");
  f.memory->set_translation(&f.replacement, &translation::resolve);
  {
    const let scope = f.memory->monitor_writes(&f.monitor);
    require(f.memory->load<std::uint32_t>({address}) == 0x55667788 &&
                f.monitor.modified && f.hook.calls == 1,
            "a two-argument setter restores opaque effects after pure mode");
  }
}

fn native_code_remapping() -> void {
  fixture f{mem::backing_policy::tracked};
  for (const let offset : std::array<std::uint64_t, 2>{0, mem::page_size}) {
    f.put(offset, offset == 0 ? 0x34030001u : 0x34030009u);
    f.put(offset + 4, 0x08180000);
    f.put(offset + 8, 0);
  }
  f.memory->set_translation(&f.original, &translation::resolve,
                            mem::translation_effect::guest_state_preserving);
  let created = jit::memory_dispatcher::create(*f.memory);
  require(created.has_value(), "create translated native code dispatcher");
  f.cpu.pc = {address};
  for (unsigned run = 0; run < 2; run++) {
    const let result = (*created)->run_slice(f.cpu, 3, 1);
    require(result && result->instructions == 3 && f.cpu.pc.value == address &&
                f.cpu.registers[3] == 1,
            "warm native code and tracked cross-slice proof");
  }
  f.memory->set_translation(&f.replacement, &translation::resolve,
                            mem::translation_effect::guest_state_preserving);
  const let remapped = (*created)->run_slice(f.cpu, 3, 1);
  require(remapped && remapped->compiled != 0 && f.cpu.registers[3] == 9,
          "translation setter revokes cached native backing before reuse");
}

} // namespace translation_proofs

namespace snapshot_proofs {

namespace mem = jit::memory;
constexpr let rx = mem::permission::read | mem::permission::execute;
fn require(bool condition, std::string_view message) -> void {
  if (!condition) {
    std::println(stderr, "source snapshot: {}", message);
    std::exit(1);
  }
}

fn put(std::span<std::byte> bytes, std::size_t offset, std::uint32_t bits)
    -> void {
  for (unsigned i = 0; i < 4; i++)
    bytes[offset + i] = static_cast<std::byte>(bits >> (24 - i * 8));
}

struct fixture {
  std::array<std::byte, 2 * mem::page_size> ram{};
  std::unique_ptr<mem::address_space> memory;
  std::unique_ptr<jit::memory_dispatcher> dispatch;
  jit::mips64::state cpu{};
  std::uint64_t translated_base;
  std::size_t patch_offset;
  unsigned callbacks = 0;
  bool armed = false, patched = false;

  fixture(std::uint64_t translated, std::size_t patch,
          mem::backing_policy policy)
      : translated_base(translated), patch_offset(patch) {
    let space = mem::address_space::create(ram, 1, policy);
    require(space.has_value(), "allocate two source pages");
    memory = std::move(*space);
    require(memory->map_ram({0x1000}, {0}, rx).has_value(),
            "map first executable page");
    memory->set_translation(this, &translate);
    let created = jit::memory_dispatcher::create(*memory);
    require(created.has_value(), "create source-checking dispatcher");
    dispatch = std::move(*created);
  }

  static fn translate(void *context, mem::virtual_address address,
                       mem::access_kind access, mem::access_size size) noexcept
      -> std::expected<mem::translated_page, mem::fault> {
    let &self = *static_cast<fixture *>(context);
    if (access != mem::access_kind::execute ||
        (address.value & ~std::uint64_t{mem::page_size - 1}) !=
            self.translated_base)
      return std::unexpected(
          mem::fault{address, mem::fault_reason::unmapped, access, size});
    self.callbacks++;
    if (self.armed && !self.patched) {
      // This models an opaque translation owner changing previously fetched
      // executable backing without a guest memory write or mapping mutation.
      put(self.ram, self.patch_offset, 0x34010007);
      self.patched = true;
    }
    return mem::translated_page{
        {mem::page_size + (address.value & (mem::page_size - 1))},
        self.ram.data() + mem::page_size,
        nullptr};
  }
};

fn cross_page_block(bool patch, mem::backing_policy policy) -> void {
  fixture f{0x2000, mem::page_size - 8, policy};
  put(f.ram, mem::page_size - 8, 0x34010001); // ORI r1,zero,1
  put(f.ram, mem::page_size - 4, 0x08000c00); // J 0x3000
  put(f.ram, mem::page_size, 0);              // Delay slot on translated page.
  f.armed = patch;
  f.cpu.pc = {0x1ff8};
  const let result = f.dispatch->run_slice(f.cpu, 3, 1);
  require(f.callbacks != 0, "delay-slot fetch resolved the later source page");
  if (patch) {
    require(f.patched && !result &&
                result.error().code == jit::dispatch_error_code::code_changed &&
                result.error().pc == 0x1ff8 && f.cpu.pc.value == 0x1ff8 &&
                f.cpu.registers[0] == 0 && f.cpu.registers[1] == 0 &&
                f.cpu.cop0.registers[9] == 0 && !f.cpu.cop0.half_cycle,
            "changed source is rejected before first native execution");
    const let retried = f.dispatch->run_slice(f.cpu, 3, 1);
    require(retried && retried->instructions == 3 && f.cpu.registers[1] == 7 &&
                f.cpu.pc.value == 0x3000,
            "a later slice recompiles and executes the fresh instruction");
  } else {
    require(result && result->instructions == 3 && f.cpu.registers[1] == 1 &&
                f.cpu.pc.value == 0x3000 && !f.patched,
            "unchanged translated source executes normally");
  }
}

} // namespace snapshot_proofs

namespace {
fn native_state_alias() -> void {
  namespace cpu = jit::mips64;
  namespace amd = jit::amd64;
  watch_proofs::fixture f;
  constexpr std::array program{cpu::instruction_word{0x34050001},
                               cpu::instruction_word{0xfc620000},
                               cpu::instruction_word{0x24a40001}};
  for (std::size_t i = 0; i < program.size(); i++)
    backing_proofs::raw_word(f.backing.extra.data() + i * 4, program[i].bits);
  const std::array source{
      std::span<const std::byte>{f.backing.extra}.first(program.size() * 4)};
  std::array<std::byte, 128 * 1024> scratch;
  std::pmr::monotonic_buffer_resource arena{scratch.data(), scratch.size(),
                                            std::pmr::null_memory_resource()};
  let translated = cpu::translate(program, {0x4000}, arena);
  backing_proofs::require(translated.has_value(),
                          "translate state-alias program");
  backing_proofs::require(jit::ir::optimize(*translated,
                                            jit::ir::fold_constants,
                                            jit::ir::simplify_identities)
                              .has_value(),
                          "optimize forwarded register reads");
  let compiler = amd::compiler::create();
  backing_proofs::require(compiler.has_value(),
                          "create state-observing compiler");
  let code = compiler->compile(*translated, arena, true);
  backing_proofs::require(code.has_value(), "compile state-alias program");
  let &state = f.backing.cpu;
  state.registers[2] = std::endian::native == std::endian::little
                           ? std::byteswap(std::uint64_t{2})
                           : 2;
  state.registers[3] = f.address(f.registers.data() + 5 * 8, true).value;
  amd::compiled_block::execution_session session{state, *f.memory};
  const let changed = code->execute(session, source);
  backing_proofs::require(
      changed && changed->instructions == 2 && changed->next_pc == 0x4008 &&
          state.registers[5] == 2 && state.registers[4] == 0,
      "native stores into aliased GPRs exit before a forwarded stale read");
  const std::array tail{program[2]};
  let resumed = cpu::translate(tail, {0x4008}, arena);
  backing_proofs::require(resumed.has_value(),
                          "translate fresh state-alias tail");
  let tail_code = compiler->compile(*resumed, arena);
  backing_proofs::require(tail_code.has_value(),
                          "compile fresh state-alias tail");
  const let result = tail_code->execute(session);
  backing_proofs::require(
      result && result->instructions == 1 && state.registers[4] == 3,
      "a later native entry reads the updated architectural state");

  constexpr std::array conditional{cpu::instruction_word{0xe0620000},
                                   cpu::instruction_word{0x24440001}};
  for (std::size_t i = 0; i < conditional.size(); i++)
    backing_proofs::raw_word(f.backing.extra.data() + i * 4,
                             conditional[i].bits);
  let sc = cpu::translate(conditional, {0x5000}, arena);
  backing_proofs::require(sc.has_value(), "translate self-aliasing SC");
  let sc_code = compiler->compile(*sc, arena, true);
  backing_proofs::require(sc_code.has_value(), "compile self-aliasing SC");
  const let target = f.address(f.registers.data() + 2 * 8, true);
  state.registers[2] = 0x12345678;
  state.registers[3] = target.value;
  state.registers[4] = 0;
  backing_proofs::require(
      f.memory->load_linked<std::uint32_t>(target).has_value(),
      "reserve the SC result register's aliased backing");
  const let stored = sc_code->execute(session, source);
  backing_proofs::require(
      stored && stored->instructions == 1 && stored->next_pc == 0x5004 &&
          state.registers[2] == 1 && state.registers[4] == 0,
      "SC commits its success result after its aliased memory write, then "
      "exits");
  const let repeated = f.memory->store_conditional<std::uint32_t>(target, 99);
  backing_proofs::require(repeated && !*repeated && state.registers[2] == 1,
                          "the self-aliasing SC clears its reservation");
}
} // namespace

fn main() -> int {
  backing_proofs::borrowed_default();
  backing_proofs::tracked_writers();
  backing_proofs::opaque_effects();
  backing_proofs::changed_mappings();
  backing_proofs::rom_policy();
  watch_proofs::independent_ranges();
  watch_proofs::boundaries();
  watch_proofs::nested_scopes();
  watch_proofs::linked_stores();
  watch_proofs::translation_effects();
  watch_proofs::mutable_code_descriptors();
  translation_proofs::ram_and_faults();
  translation_proofs::mmio_effects();
  translation_proofs::setter_invalidates();
  translation_proofs::native_code_remapping();
  for (const auto policy : {jit::memory::backing_policy::borrowed,
                            jit::memory::backing_policy::tracked}) {
    snapshot_proofs::cross_page_block(false, policy);
    snapshot_proofs::cross_page_block(true, policy);
  }
  native_state_alias();
  std::println("ownership: borrowed defaults, tracked CPU/DMA/alias writes, "
               "opaque and preserving effects, monitor/state ranges, mapping "
               "changes and fresh first execution passed");
}
