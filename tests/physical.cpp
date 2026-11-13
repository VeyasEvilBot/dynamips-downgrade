#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <print>
#include <string_view>
#include <utility>
#include <span>

#include <dynamips/macros.h>

import jit.memory;
import jit.physical;

namespace {
namespace mem = jit::memory;
namespace physical = jit::physical;
constexpr let rwx =
    mem::permission::read | mem::permission::write | mem::permission::execute;

fn require(bool condition, std::string_view message) -> void {
  if (!condition) {
    std::println(stderr, "physical bus test: {}", message);
    std::exit(1);
  }
}

fn regions_and_permissions() -> void {
  std::array<std::byte, 2 * mem::page_size> ram{};
  std::array<std::byte, mem::page_size> rom{};
  mem::device endpoint{nullptr, nullptr, nullptr};
  const std::array regions{
      physical::region{{0}, ram.size(), ram, nullptr, rwx},
      physical::region{{0x10000},
                       rom.size(),
                       rom,
                       nullptr,
                       mem::permission::read | mem::permission::execute},
      physical::region{{0x20000},
                       mem::page_size,
                       {},
                       &endpoint,
                       mem::permission::read | mem::permission::write},
      physical::region{{0x30000}, ram.size(), ram, nullptr, rwx}};
  let space = physical::bus::create(regions);
  require(space.has_value(), "create RAM, ROM and MMIO physical regions");
  for (const let access : {mem::access_kind::read, mem::access_kind::write,
                           mem::access_kind::execute}) {
    const let resolved =
        space->resolve({mem::page_size + 12}, access, mem::access_size::word);
    require(resolved && resolved->physical.value == mem::page_size + 12 &&
                resolved->host_page == ram.data() + mem::page_size &&
                !resolved->target,
            "physical resolution retains the exact address and returns its "
            "backing page");
  }
  const let cached_alias =
      space->resolve({12}, mem::access_kind::read, mem::access_size::word);
  const let uncached_alias =
      space->resolve({0x3000c}, mem::access_kind::read, mem::access_size::word);
  require(cached_alias && uncached_alias &&
              cached_alias->host_page == uncached_alias->host_page,
          "CPU aliases resolve to the same physical backing without installed "
          "virtual maps");
  const let rom_read = space->resolve({0x10008}, mem::access_kind::execute,
                                      mem::access_size::word);
  const let rom_write = space->resolve({0x10008}, mem::access_kind::write,
                                       mem::access_size::word);
  require(rom_read && rom_read->host_page == rom.data() && !rom_write &&
              rom_write.error().reason == mem::fault_reason::permission,
          "ROM is executable and read-only");
  const let mmio = space->resolve({0x20010}, mem::access_kind::write,
                                  mem::access_size::word);
  const let execute_mmio = space->resolve({0x20010}, mem::access_kind::execute,
                                          mem::access_size::word);
  require(mmio && mmio->physical.value == 0x20010 && !mmio->host_page &&
              mmio->target == &endpoint && !execute_mmio &&
              execute_mmio.error().reason == mem::fault_reason::permission,
          "MMIO resolution selects the device and never permits execution");
  require(space->resolve({ram.size() - 8}, mem::access_kind::read,
                         mem::access_size::doubleword)
                  .has_value() &&
              !space->resolve({ram.size() - 1}, mem::access_kind::read,
                              mem::access_size::doubleword) &&
              !space->resolve({ram.size()}, mem::access_kind::read,
                              mem::access_size::byte),
          "complete accesses must remain inside one physical region");
  const let hole =
      space->resolve({0x40000}, mem::access_kind::read, mem::access_size::word);
  require(!hole && hole.error().reason == mem::fault_reason::bus &&
              hole.error().address.value == 0x40000,
          "missing physical hardware remains a bus failure");
  let moved = std::move(*space);
  require(moved.resolve({0}, mem::access_kind::read, mem::access_size::word)
                  ->host_page == ram.data(),
          "moving the bus preserves borrowed backing identities");
}

fn invalid_regions() -> void {
  std::array<std::byte, mem::page_size> ram{};
  mem::device endpoint{nullptr, nullptr, nullptr};
  const let rejected = [&](physical::region region,
                           physical::error_code reason) {
    const std::array regions{region};
    const let created = physical::bus::create(regions);
    require(!created && created.error().code == reason &&
                created.error().index == 0,
            "reject malformed physical region with its descriptor index");
  };
  rejected({{0}, 0, ram}, physical::error_code::empty_region);
  rejected({{1}, mem::page_size, ram}, physical::error_code::misaligned_region);
  rejected({{0}, mem::page_size - 1, ram},
           physical::error_code::misaligned_region);
  rejected({{UINT64_MAX - mem::page_size + 1}, mem::page_size, ram},
           physical::error_code::overflowing_region);
  rejected({{0}, 2 * mem::page_size, ram},
           physical::error_code::invalid_backing);
  rejected({{0}, mem::page_size, {}, nullptr},
           physical::error_code::invalid_backing);
  rejected({{0}, mem::page_size, ram, &endpoint},
           physical::error_code::invalid_backing);
  rejected({{0}, mem::page_size, {}, &endpoint, rwx},
           physical::error_code::invalid_permissions);
  rejected({{0}, mem::page_size, ram, nullptr, static_cast<mem::permission>(0)},
           physical::error_code::invalid_permissions);
  const std::array overlap{physical::region{{0}, mem::page_size, ram},
                           physical::region{{0}, mem::page_size, ram}};
  const let collision = physical::bus::create(overlap);
  require(!collision &&
              collision.error().code == physical::error_code::overlap &&
              collision.error().index == 1,
          "overlapping physical responders are rejected before publication");
  std::array<physical::region, physical::bus::capacity + 1> too_many{};
  const let overflow = physical::bus::create(too_many);
  require(!overflow &&
              overflow.error().code == physical::error_code::capacity_exhausted,
          "physical region storage remains bounded");
  require(physical::bus::create({}).has_value(),
          "an empty physical bus is valid");
}
fn coherent_writes() -> void {
  std::array<std::byte, 2 * mem::page_size> ram{};
  let space = mem::address_space::create(ram, 2);
  require(space && (*space)->map_ram({0x1000}, {0}, rwx).has_value() &&
              (*space)->map_ram({0x3000}, {0}, rwx).has_value(),
          "map physical-write aliases");
  const let first = (*space)->stamp(std::span{ram}.first(4));
  const let second = (*space)->stamp(std::span{ram}.subspan(mem::page_size, 4));
  const std::array code{std::span<const std::byte>{ram}.first(4)};
  mem::write_monitor monitor{code};
  const std::array bytes{std::byte{1}, std::byte{2}, std::byte{3},
                         std::byte{4}};
  const let scope = (*space)->monitor_writes(&monitor);
  require((*space)->load_linked<std::uint32_t>({0x3000}).has_value(),
          "reserve physical alias");
  const let generation = (*space)->mapping_generation();
  require((*space)->write_physical({0}, bytes).has_value() &&
              monitor.modified && first.value() == 1 &&
              (*space)->store_conditional<std::uint32_t>({0x1000}, 9) ==
                  false &&
              (*space)->load<std::uint32_t>({0x3000}) == 0x01020304 &&
              (*space)->mapping_generation() == generation,
          "DMA-style writes notify code guards and LL/SC without invalidating "
          "translations");
  monitor.modified = false;
  require((*space)->write_physical({mem::page_size - 2}, bytes).has_value() &&
              first.value() == 2 && second.value() == 1 && !monitor.modified,
          "cross-page physical writes update every affected RAM page version");
  const let old = ram.back();
  require(!(*space)->write_physical({ram.size() - 1}, bytes) &&
              ram.back() == old && second.value() == 1,
          "out-of-range DMA writes fail before partial mutation");
}

} // namespace

fn main() -> int {
  regions_and_permissions();
  invalid_regions();
  coherent_writes();
  std::println("physical bus: regions, bounds, permissions and backing "
               "identities passed");
}
