#include <array>
#include <bit>
#include <cstddef>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <print>
#include <span>
#include <string_view>

#include <dynamips/macros.h>

import jit.memory;
import jit.fifo;
import jit.uart;

namespace {
namespace mem = jit::memory;
constexpr let rw = mem::permission::read | mem::permission::write;

fn require(bool condition, std::string_view message) -> void {
  if (!condition) {
    std::println(stderr, "memory test: {}", message);
    std::exit(1);
  }
}

fn executable_write_proofs() -> void {
  std::array<std::byte, 3 * mem::page_size> backing{};
  let memory = mem::address_space::create(backing, 4);
  require(memory && (*memory)->map_ram({0x1000}, {0}, rw).has_value() &&
              (*memory)->map_ram({0x2000}, {mem::page_size}, rw).has_value() &&
              (*memory)->map_ram({0x4000}, {0}, rw).has_value(),
          "map versioned RAM aliases");
  const let spanning =
      (*memory)->stamp(std::span{backing}.subspan(mem::page_size - 2, 4));
  const let first = (*memory)->stamp(std::span{backing}.first(4));
  const let second =
      (*memory)->stamp(std::span{backing}.subspan(mem::page_size, 4));
  require(spanning.value() == 0 && first.value() == 0 && second.value() == 0,
          "new backing stamps start at zero");
  const std::array code{std::span<const std::byte>{backing}.subspan(4, 4)};
  mem::write_monitor monitor{code};
  {
    const let scope = (*memory)->monitor_writes(&monitor);
    require((*memory)->store<std::uint32_t>({0x4100}, 7).has_value() &&
                !monitor.modified && spanning.value() == 1 &&
                first.value() == 1 && second.value() == 0,
            "data alias stores update page versions without dirtying unrelated "
            "code");
    require((*memory)->store<std::uint32_t>({0x2000}, 8).has_value() &&
                spanning.value() == 2 && second.value() == 1,
            "unaligned backing stamps include every intersected RAM page");
    require((*memory)
                    ->store_bytes({0x4004}, std::as_bytes(std::span{"abcd", 4}))
                    .has_value() &&
                monitor.modified && first.value() == 2,
            "partial alias stores notify executable ranges and page versions");
  }
  const let before = first.value();
  require((*memory)->store_conditional<std::uint32_t>({0x1000}, 9) == false &&
              first.value() == before,
          "failed conditional writes leave executable versions unchanged");
  require((*memory)->load_linked<std::uint32_t>({0x1000}).has_value() &&
              (*memory)->store_conditional<std::uint32_t>({0x4000}, 9) ==
                  true &&
              first.value() == before + 1,
          "successful conditional writes through an alias update executable "
          "versions");
}

struct device_state {
  unsigned reads = 0;
  unsigned writes = 0;
  std::uint64_t value = 0x12345678;
  std::uint64_t last_address = 0;
};

fn device_read(void *context, mem::device_access access) noexcept
    -> std::expected<std::uint64_t, mem::device_error> {
  let &state = *static_cast<device_state *>(context);
  state.reads++;
  state.last_address = access.address.value;
  if (access.size != mem::access_size::word)
    return std::unexpected(mem::device_error::unsupported_size);
  return access.order == std::endian::little
             ? std::byteswap(static_cast<std::uint32_t>(state.value))
             : state.value;
}
fn device_write(void *context, mem::device_access access,
                  std::uint64_t value) noexcept
    -> std::expected<void, mem::device_error> {
  let &state = *static_cast<device_state *>(context);
  state.writes++;
  state.last_address = access.address.value;
  if (access.size != mem::access_size::word)
    return std::unexpected(mem::device_error::bus);
  state.value = access.order == std::endian::little
                    ? std::byteswap(static_cast<std::uint32_t>(value))
                    : value;
  return {};
}

template <std::unsigned_integral T>
fn roundtrip(mem::address_space &memory, T value) -> void {
  require(memory.store<T>({32}, value).has_value(), "big endian store");
  const let loaded = memory.load<T>({32});
  require(loaded && *loaded == value, "big endian load");
  const let reversed = memory.load<T, std::endian::little>({32});
  require(reversed && *reversed == std::byteswap(value), "explicit byte order");
  require(memory.store<T, std::endian::little>({32}, value).has_value(),
          "little endian store");
  require(memory.load<T, std::endian::little>({32}) == value,
          "little endian load");
}
} // namespace

fn fifo_io() -> void {
  jit::byte_fifo<8> queue;
  constexpr std::array first{std::byte{1}, std::byte{2}, std::byte{3},
                             std::byte{4}, std::byte{5}, std::byte{6}};
  require(queue.write(first) == 6 && queue.consume(4).has_value(),
          "FIFO prefix consumption");
  constexpr std::array second{std::byte{7},  std::byte{8},  std::byte{9},
                              std::byte{10}, std::byte{11}, std::byte{12}};
  require(queue.write(second) == 6 && !queue.push(std::byte{13}),
          "bounded ring wrap and full detection");
  std::array<std::byte, 8> collected;
  std::size_t written = 0;
  unsigned calls = 0;
  const let partial = queue.drain([&](std::span<const std::byte> bytes) noexcept
                                      -> std::expected<std::size_t, int> {
    calls++;
    if (calls == 2)
      return std::unexpected(7);
    for (std::size_t i = 0; i < 2; i++)
      collected[written++] = bytes[i];
    return 2;
  });
  require(!partial && partial.error().transferred == 2 && queue.size() == 6,
          "partial drain retains unwritten suffix on error");
  const let stalled =
      queue.drain([](std::span<const std::byte>) noexcept
                      -> std::expected<std::size_t, int> { return 0; });
  require(stalled && *stalled == 0 && queue.size() == 6,
          "zero progress yields without spinning");
  const let invalid = queue.drain(
      [](std::span<const std::byte> bytes) noexcept
          -> std::expected<std::size_t, int> { return bytes.size() + 1; });
  require(!invalid && queue.size() == 6 && !queue.consume(9),
          "reject invalid sink counts without dropping data");
  calls = 0;
  const let drained = queue.drain([&](std::span<const std::byte> bytes) noexcept
                                      -> std::expected<std::size_t, int> {
    calls++;
    for (const let byte : bytes)
      collected[written++] = byte;
    return bytes.size();
  });
  require(drained && *drained == 6 && calls == 2 && queue.size() == 0 &&
              !queue.pop(),
          "wrapped FIFO drains in contiguous batches");
  for (std::size_t i = 0; i < collected.size(); i++)
    require(collected[i] == static_cast<std::byte>(i + 5),
            "FIFO order survives partial writes and wrap");

  jit::uart uart{{0x100000}};
  let space = mem::address_space::create({}, 1);
  require(space &&
              (*space)
                  ->map_device({0x8000}, {0x100000}, uart.endpoint(),
                               mem::permission::read | mem::permission::write)
                  .has_value(),
          "map queued UART");
  constexpr std::array input{std::byte{65}, std::byte{66}};
  require(uart.receive(input) == 2 &&
              (*space)->load<std::uint32_t>({0x8004}) == 3,
          "RX FIFO advertises input readiness");
  require((*space)->load<std::uint32_t>({0x8008}) == 65 &&
              (*space)->load<std::uint32_t>({0x8008}) == 66 &&
              (*space)->load<std::uint32_t>({0x8004}) == 1 &&
              !(*space)->load<std::uint32_t>({0x8008}),
          "RX reads consume bytes and report empty input");
  for (std::size_t i = 0; i < jit::uart::capacity; i++)
    require((*space)
                ->store<std::uint32_t>({0x8000},
                                       static_cast<std::uint32_t>(i & 255))
                .has_value(),
            "fill TX FIFO");
  const let drain = uart.drain(
      [](std::span<const std::byte> bytes) noexcept
          -> std::expected<std::size_t, int> { return bytes.size(); });
  require(drain && *drain == jit::uart::capacity && uart.pending() == 0 &&
              (*space)->load<std::uint32_t>({0x8004}) == 1,
          "draining restores TX readiness");
}

fn translated_cache() -> void {
  std::array<std::byte, mem::page_size> ram{};
  let space = mem::address_space::create(ram, 2);
  require(space.has_value(), "create translated address space");

  struct state {
    std::byte *page;
    unsigned calls = 0;
    bool writable = true;
  } backing{ram.data()};

  (*space)->set_translation(
      &backing,
      [](void *context, mem::virtual_address address, mem::access_kind access,
         mem::access_size size) noexcept
          -> std::expected<mem::translated_page, mem::fault> {
        let &value = *static_cast<state *>(context);
        value.calls++;
        if (access == mem::access_kind::write && !value.writable)
          return std::unexpected(
              mem::fault{address, mem::fault_reason::permission, access, size});
        return mem::translated_page{
            {address.value & (mem::page_size - 1)}, value.page, nullptr};
      });
  require((*space)->store<std::uint32_t>({0x60000000}, 42).has_value() &&
              (*space)->load<std::uint32_t>({0x60000000}) == 42 &&
              backing.calls == 2,
          "translation fills separate read and write caches");
  require((*space)->load<std::uint32_t>({0x60000000}) == 42 &&
              (*space)->store<std::uint32_t>({0x60000004}, 43).has_value() &&
              backing.calls == 2,
          "warm mapped accesses bypass translation callbacks");
  backing.writable = false;
  (*space)->invalidate();
  const let denied = (*space)->store<std::uint32_t>({0x60000000}, 99);
  require(!denied && denied.error().reason == mem::fault_reason::permission &&
              (*space)->load<std::uint32_t>({0x60000000}) == 42,
          "invalidation rechecks write permissions without modifying RAM");
  backing.writable = true;
  const let rx = mem::permission::read | mem::permission::execute;
  require((*space)->store<std::uint32_t>({0x60000000}, 42).has_value() &&
              (*space)->map_ram({0x1000}, {0}, rx).has_value() &&
              (*space)->seal_executable_pages().has_value(),
          "seal code after warming a translated writable alias");
  const let alias = (*space)->store<std::uint32_t>({0x60000000}, 99);
  require(!alias && alias.error().reason == mem::fault_reason::permission &&
              (*space)->fetch<std::uint32_t>({0x1000}) == 42 &&
              !(*space)->fetch<std::uint32_t>({0x60000000}),
          "sealed code rejects translated writes and requires static execution "
          "mappings");
  require((*space)->unmap({0x1000}).has_value() &&
              (*space)->store<std::uint32_t>({0x60000000}, 43).has_value() &&
              (*space)->map_ram({0x1000}, {0}, rx).has_value() &&
              !(*space)->store<std::uint32_t>({0x60000000}, 99) &&
              (*space)->fetch<std::uint32_t>({0x1000}) == 43,
          "new sealed mappings invalidate warm translated aliases");
  const std::array remove{
      mem::mapping_update{mem::update_kind::unmap, {0x1000}}};
  const let removed = (*space)->apply_updates(remove);
  require(removed && removed->full_flush &&
              (*space)->store<std::uint32_t>({0x60000000}, 44).has_value(),
          "batched unmapping flushes translated aliases");
  const std::array install{
      mem::mapping_update{mem::update_kind::ram, {0x2000}, {0}, rx}};
  const let installed = (*space)->apply_updates(install);
  require(installed && installed->full_flush &&
              !(*space)->store<std::uint32_t>({0x60000000}, 99) &&
              (*space)->fetch<std::uint32_t>({0x2000}) == 44,
          "batched code mappings protect backing after a full cache flush");
}

fn reservations() -> void {
  std::array<std::byte, mem::page_size> ram{};
  let space = mem::address_space::create(ram, 4);
  require(
      space && (*space)->map_ram({0x1000}, {0}, rw).has_value() &&
          (*space)->map_ram({0x2000}, {0}, rw).has_value() &&
          (*space)->map_ram({0x3000}, {0}, mem::permission::read).has_value(),
      "map reservation aliases");
  let &memory = **space;
  require(memory.store_conditional<std::uint32_t>({0x1000}, 1) == false,
          "conditional store without reservation fails");
  require(memory.store<std::uint32_t>({0x1000}, 0x80000000).has_value() &&
              memory.load_linked<std::uint32_t>({0x1000}) == 0x80000000 &&
              memory.store_conditional<std::uint32_t>({0x2000}, 42) == true &&
              memory.load<std::uint32_t>({0x1000}) == 42 &&
              memory.store_conditional<std::uint32_t>({0x1000}, 99) == false,
          "reservation follows physical aliases and succeeds only once");
  require(memory.load_linked<std::uint32_t>({0x1000}).has_value() &&
              memory.store<std::uint32_t>({0x1008}, 1).has_value() &&
              memory.store_conditional<std::uint32_t>({0x1000}, 99) == false,
          "ordinary writes conservatively break reservations");
  const std::array byte{std::byte{1}};
  require(memory.load_linked<std::uint32_t>({0x1000}).has_value() &&
              memory.store_bytes({0x1008}, byte).has_value() &&
              memory.store_conditional<std::uint32_t>({0x1000}, 99) == false,
          "partial writes break reservations");
  require(memory.load_linked<std::uint32_t>({0x1000}).has_value(),
          "reserve before flush");
  memory.invalidate();
  require(memory.store_conditional<std::uint32_t>({0x1000}, 99) == false,
          "translation flush breaks reservation");
  require(memory.load_linked<std::uint32_t>({0x1000}).has_value() &&
              memory.protect({0x2000}, mem::permission::read).has_value() &&
              memory.store_conditional<std::uint32_t>({0x1000}, 99) == false,
          "mapping edits break reservation");
  require(memory.load_linked<std::uint32_t>({0x1000}).has_value() &&
              !memory.load_linked<std::uint32_t>({0x1001}) &&
              memory.store_conditional<std::uint32_t>({0x1000}, 99) == false,
          "faulting linked load discards prior reservation");
  require(memory.load_linked<std::uint32_t>({0x1000}).has_value() &&
              memory.store_conditional<std::uint64_t>({0x1000}, 99) == false,
          "reservation width must match");
  require(memory.load_linked<std::uint32_t>({0x3000}).has_value(),
          "reserve read-only alias");
  const let denied = memory.store_conditional<std::uint32_t>({0x3000}, 99);
  require(!denied && denied.error().reason == mem::fault_reason::permission &&
              memory.store_conditional<std::uint32_t>({0x1000}, 99) == false,
          "conditional stores validate permissions and consume reservation on "
          "fault");
  require(memory.load_linked<std::uint64_t>({0x1000}).has_value() &&
              memory.store_conditional<std::uint64_t>(
                  {0x1000}, 0x1234567887654321ULL) == true,
          "doubleword reservations preserve all bits");
  unsigned callbacks = 0;
  mem::device device{
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
  require(memory.map_device({0x4000}, {0x100000}, device, rw).has_value() &&
              !memory.load_linked<std::uint32_t>({0x4000}) &&
              !memory.store_conditional<std::uint32_t>({0x4000}, 1) &&
              callbacks == 0,
          "linked device accesses fail without MMIO callbacks");
}

fn main() -> int {
  executable_write_proofs();
  reservations();
  translated_cache();
  fifo_io();
  std::array<std::byte, 3 * mem::page_size> ram{};
  let created = mem::address_space::create(ram, 8);
  require(created.has_value(), "create bounded address space");
  let &memory = **created;
  require(memory.map_ram({0}, {0}, rw).has_value(), "map first RAM page");
  const std::array bytes{std::byte{1}, std::byte{2}, std::byte{3}};
  require(
      memory.store_bytes({8}, bytes).has_value() &&
          memory.store_bytes({9}, std::span{ram}.subspan(8, 3)).has_value() &&
          ram[8] == std::byte{1} && ram[9] == std::byte{1} &&
          ram[10] == std::byte{2} && ram[11] == std::byte{3},
      "byte writes support overlapping borrowed input");
  const let unchanged = ram;
  const let crossing = memory.store_bytes({mem::page_size - 2}, bytes);
  require(!crossing &&
              crossing.error().reason == mem::fault_reason::unsupported_size &&
              ram == unchanged,
          "single-page byte writes reject page crossing before changing RAM");
  roundtrip(memory, std::uint8_t{0xab});
  roundtrip(memory, std::uint16_t{0xabcd});
  roundtrip(memory, std::uint32_t{0x12345678});
  roundtrip(memory, std::uint64_t{0x0123456789abcdef});
  require(memory.store<std::uint32_t>({0}, 0x12345678).has_value(),
          "store known bytes");
  require(ram[0] == std::byte{0x12} && ram[3] == std::byte{0x78},
          "RAM remains in guest byte order");
  require(memory.store<std::uint64_t>({mem::page_size - 8}, 77).has_value(),
          "page-end access");
  const let before = ram;
  const let bad_store = memory.store<std::uint64_t>({mem::page_size - 4}, 99);
  require(!bad_store &&
              bad_store.error().reason == mem::fault_reason::misaligned &&
              ram == before,
          "misaligned store has no partial effect");
  const let absent = memory.load<std::uint32_t>({mem::page_size});
  require(!absent && absent.error().reason == mem::fault_reason::unmapped &&
              absent.error().address.value == mem::page_size &&
              absent.error().access == mem::access_kind::read &&
              absent.error().size == mem::access_size::word,
          "unmapped fault retains access details");
  require(!memory.fetch<std::uint32_t>({0}),
          "read cache cannot authorize execution");

  constexpr std::uint64_t collision =
      mem::address_space::first_level_entries * mem::page_size;
  require(memory.map_ram({collision}, {mem::page_size}, rw).has_value(),
          "map L1 collision");
  require(memory.store<std::uint32_t>({collision}, 22).has_value(),
          "write second physical page");
  for (unsigned i = 0; i < 100; i++) {
    require(memory.load<std::uint32_t>({0}) == 0x12345678,
            "L2 refill retains first page");
    require(memory.load<std::uint32_t>({collision}) == 22,
            "L2 refill retains colliding page");
  }
  require(memory.protect({0}, mem::permission::execute).has_value(),
          "change cached permissions");
  const let denied = memory.load<std::uint32_t>({0});
  require(!denied && denied.error().reason == mem::fault_reason::permission,
          "revoke L2 read permission");
  require(!memory.store<std::uint32_t>({0}, 1),
          "revoke cached write permission");
  require(memory.fetch<std::uint32_t>({0}) == 0x12345678,
          "execute-only mapping");
  require(memory.protect({collision}, mem::permission::execute).has_value() &&
              memory.fetch<std::uint32_t>({collision}) == 22,
          "evict execute entry into L2");
  require(memory.protect({0}, rw).has_value() &&
              !memory.fetch<std::uint32_t>({0}),
          "revoking execute permission invalidates L2");
  require(memory.protect({collision}, rw).has_value(), "restore data mapping");
  require(memory.map_ram({0}, {mem::page_size}, rw).has_value(),
          "replace cached physical backing");
  require(memory.load<std::uint32_t>({0}) == 22,
          "remap invalidates all cache levels");
  require(!memory.fetch<std::uint32_t>({0}),
          "remap cannot retain execute authorization");
  require(memory.load<std::uint32_t>({collision}) == 22,
          "warm L1 collision before unmap");
  require(memory.unmap({0}).has_value(), "unmap cached page");
  require(!memory.load<std::uint32_t>({0}), "unmap invalidates L2");
  memory.invalidate();
  require(memory.load<std::uint32_t>({collision}) == 22,
          "full flush refills correctly");

  constexpr std::uint64_t both_collision =
      (mem::address_space::second_level_entries *
           mem::address_space::first_level_entries -
       mem::address_space::first_level_entries) *
      mem::page_size;
  require(memory.map_ram({both_collision + collision}, {2 * mem::page_size}, rw)
              .has_value(),
          "map L2 collision");
  require(
      memory.store<std::uint32_t>({both_collision + collision}, 33).has_value(),
      "write L2 collision");
  for (unsigned i = 0; i < 20; i++) {
    require(memory.load<std::uint32_t>({collision}) == 22,
            "authoritative refill after L2 eviction");
    require(memory.load<std::uint32_t>({both_collision + collision}) == 33,
            "L2 conflict never aliases pages");
  }
  constexpr std::uint64_t last_page = UINT64_MAX - mem::page_size + 1;
  require(memory.map_ram({last_page}, {0}, rw).has_value(),
          "map highest virtual page");
  require(memory.load<std::uint64_t>({UINT64_MAX - 7}) == 77,
          "highest aligned address does not wrap");
  require(!memory.map_ram({1}, {0}, rw), "reject unaligned virtual page");
  require(!memory.map_ram({0}, {1}, rw), "reject unaligned physical page");
  require(!memory.map_ram({0}, {3 * mem::page_size}, rw),
          "reject physical range outside RAM");
  require(!memory.map_ram({0}, {UINT64_MAX}, rw),
          "reject overflowing physical range");
  require(!memory.map_ram({0}, {0}, static_cast<mem::permission>(0)),
          "reject empty permissions");

  device_state state;
  mem::device device{&state, device_read, device_write};
  require(!memory.map_device({0x400000}, {0}, device, rw),
          "device cannot overlap physical RAM");
  require(!memory.map_device({0x400000}, {0xdead000}, device,
                             mem::permission::execute),
          "device execution is rejected at registration");
  require(memory.map_device({0x400000}, {0xdead000}, device, rw).has_value(),
          "map MMIO");
  require(memory.load<std::uint32_t>({0x400004}) == 0x12345678,
          "MMIO returns numeric values");
  require(memory.load<std::uint32_t>({0x400004}) == 0x12345678 &&
              state.reads == 2,
          "MMIO reads are never bypassed by cached RAM pointers");
  require(state.last_address == 0xdead004, "MMIO receives physical address");
  require(memory.store<std::uint32_t>({0x400004}, 0xabcdef01).has_value() &&
              state.value == 0xabcdef01,
          "MMIO write is one width-preserving numeric transaction");
  require(memory.load<std::uint32_t, std::endian::little>({0x400004}) ==
              0x01efcdab,
          "MMIO receives load byte order");
  require(
      memory.store<std::uint32_t, std::endian::little>({0x400004}, 0x01020304)
              .has_value() &&
          state.value == 0x04030201,
      "MMIO receives store byte order");
  const let unsupported = memory.load<std::uint16_t>({0x400004});
  require(!unsupported &&
              unsupported.error().reason == mem::fault_reason::unsupported_size,
          "typed device width failure");
  const let bus = memory.store<std::uint16_t>({0x400004}, 1);
  require(!bus && bus.error().reason == mem::fault_reason::bus,
          "typed device bus failure");
  const let count = state.reads;
  require(!memory.load<std::uint32_t>({0x400001}) && state.reads == count,
          "misalignment checked before MMIO");
  require(memory.map_ram({0x400000}, {mem::page_size}, rw).has_value(),
          "replace device mapping");
  require(memory.load<std::uint32_t>({0x400000}) == 22 && state.reads == count,
          "device-to-RAM remap");
  require(memory.map_device({0x400000}, {0xdead000}, device, rw).has_value(),
          "replace cached RAM with MMIO");
  require(memory.load<std::uint32_t>({0x400000}) == state.value &&
              state.reads == count + 1,
          "RAM-to-device remap invalidates cache");
  require(memory.map_device({0x500000}, {0xdead000}, device, rw).has_value(),
          "same device may have virtual aliases");
  mem::device other_device{&state, device_read, device_write};
  const let conflict =
      memory.map_device({0x600000}, {0xdead000}, other_device, rw);
  require(!conflict &&
              conflict.error() == mem::mapping_error::physical_conflict,
          "different devices cannot share a physical page");
  require(memory.load<std::uint32_t>({0x500000}) == state.value &&
              memory.load<std::uint32_t>({0x400000}) == state.value,
          "rejected mapping leaves device aliases usable");
  let bounded = mem::address_space::create(ram, 1);
  require(bounded && (*bounded)->map_ram({0}, {0}, rw).has_value(),
          "fill bounded map");
  const let full = (*bounded)->map_ram({mem::page_size}, {0}, rw);
  require(!full && full.error() == mem::mapping_error::capacity_exhausted,
          "mapping never grows past startup capacity");
  std::println("memory: RAM, endian, cache levels, invalidation, permissions "
               "and MMIO checks passed");
  let batched = mem::address_space::create(ram, 80);
  require(batched.has_value(), "create batch memory");
  std::array<mem::mapping_update, 70> updates;
  for (std::size_t i = 0; i < updates.size(); i++)
    updates[i] = {mem::update_kind::ram, {i * mem::page_size}, {0}, rw};
  const let initial_generation = (*batched)->mapping_generation();
  const let batch = (*batched)->apply_updates(updates);
  require(batch && batch->applied == 70 && batch->full_flush &&
              (*batched)->mapping_generation() == initial_generation + 1,
          "large batch uses one bounded full flush");
  require((*batched)->store<std::uint32_t>({0}, 7).has_value() &&
              (*batched)->load<std::uint32_t>({0}) == 7,
          "warm batch cache");
  constexpr std::array repeated{
      mem::mapping_update{
          mem::update_kind::protect, {0}, {}, mem::permission::read},
      mem::mapping_update{mem::update_kind::protect, {0}, {}, rw},
      mem::mapping_update{
          mem::update_kind::protect, {0}, {}, mem::permission::read}};
  const let generation = (*batched)->mapping_generation();
  const let deduplicated = (*batched)->apply_updates(repeated);
  require(deduplicated && deduplicated->invalidated_pages == 1 &&
              !deduplicated->full_flush &&
              (*batched)->mapping_generation() == generation + 1 &&
              !(*batched)->store<std::uint32_t>({0}, 8),
          "duplicate invalidations are coalesced before access");
  constexpr std::array failing{
      mem::mapping_update{mem::update_kind::unmap, {0}},
      mem::mapping_update{mem::update_kind::unmap, {1}}};
  const let failed_batch = (*batched)->apply_updates(failing);
  require(!failed_batch && failed_batch.error().index == 1 &&
              !(*batched)->load<std::uint32_t>({0}),
          "failed batch still flushes earlier edits");
}
