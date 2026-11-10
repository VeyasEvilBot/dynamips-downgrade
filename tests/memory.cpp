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

namespace {
namespace mem = jit::memory;
constexpr let rw = mem::permission::read | mem::permission::write;

fn require(bool condition, std::string_view message) -> void {
  if (!condition) {
    std::println(stderr, "memory test: {}", message);
    std::exit(1);
  }
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

fn main() -> int {
  std::array<std::byte, 3 * mem::page_size> ram{};
  let created = mem::address_space::create(ram, 8);
  require(created.has_value(), "create bounded address space");
  let &memory = **created;
  require(memory.map_ram({0}, {0}, rw).has_value(), "map first RAM page");
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
}
