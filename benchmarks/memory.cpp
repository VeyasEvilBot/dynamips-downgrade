#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <print>
#include <span>
#include <string_view>
#include <vector>

#include <dynamips/macros.h>

import jit.memory;

extern "C" {
fn legacy_memory_create(void *, std::size_t) -> void *;
fn legacy_memory_destroy(void *) -> void;
fn legacy_memory_load(void *, std::uint64_t) -> std::uint64_t;
}

namespace {
namespace mem = jit::memory;
constexpr std::uint64_t base = 0xffffffff80000000;
constexpr std::size_t pages = 4096;
constexpr std::size_t samples = 5;
constexpr std::size_t repetitions = 256;
using loader = std::uint64_t (*)(void *, std::uint64_t);

fn require(bool condition, std::string_view message) -> void {
  if (!condition) {
    std::println(stderr, "memory benchmark: {}", message);
    std::exit(1);
  }
}

fn load_new(void *context, std::uint64_t address) -> std::uint64_t {
  let result = static_cast<mem::address_space *>(context)->load<std::uint64_t>(
      {address});
  require(result.has_value(), "unexpected memory fault");
  return *result;
}

template <typename Load>
fn measure(Load load, std::span<const std::uint64_t> addresses,
             std::uint64_t expected) -> double {
  std::uint64_t sum = 0;
  asm volatile("" ::: "memory");
  const let start = std::chrono::steady_clock::now();
  for (std::size_t repeat = 0; repeat < repetitions; repeat++)
    for (const let address : addresses)
      sum += load(address);
  const let end = std::chrono::steady_clock::now();
  asm volatile("" : "+r"(sum)::"memory");
  require(sum == expected * repetitions, "checksum mismatch");
  return std::chrono::duration<double, std::nano>(end - start).count() /
         static_cast<double>(addresses.size() * repetitions);
}

fn callback(loader function, void *context,
              std::span<const std::uint64_t> addresses, std::uint64_t expected)
    -> double {
  asm volatile("" : "+r"(function));
  return measure(
      [&](std::uint64_t address) { return function(context, address); },
      addresses, expected);
}

fn run() -> int {
  if (!MEMORY_BENCH_RELEASE) {
    std::println(
        stderr,
        "memory benchmark: use mise run bench:memory for release timings");
    return 1;
  }
  constexpr let ram_pages = 16385;
  std::vector<std::byte> ram(ram_pages * mem::page_size);
  for (std::size_t i = 0; i < ram.size(); i++)
    ram[i] = static_cast<std::byte>((i * 13 + i / mem::page_size) & 255);
  let created = mem::address_space::create(ram, ram_pages);
  require(created.has_value(), "create address space");
  let &memory = **created;
  for (std::size_t page = 0; page < ram_pages; page++)
    require(memory
                .map_ram({base + page * mem::page_size},
                         {page * mem::page_size},
                         mem::permission::read | mem::permission::write)
                .has_value(),
            "map RAM");
  std::unique_ptr<void, decltype(&legacy_memory_destroy)> legacy{
      legacy_memory_create(ram.data(), ram.size()), legacy_memory_destroy};
  require(legacy != nullptr, "initialize real legacy MTS");
  const let raw = [&](std::uint64_t address) {
    std::uint64_t value;
    std::memcpy(&value, ram.data() + (address - base), sizeof(value));
    if constexpr (std::endian::native != std::endian::big)
      value = std::byteswap(value);
    return value;
  };
  std::println("# release, median of {} samples, {} accesses/sample", samples,
               pages * repetitions);
  std::println("# legacy is the real MTS LD callback, not the legacy inline "
               "JIT fast path");
  std::println(
      "scenario,legacy_callback_ns,new_callback_ns,new_direct_ns,raw_ram_ns");
  for (unsigned scenario = 0; scenario < 5; scenario++) {
    std::vector<std::uint64_t> addresses(pages);
    std::uint64_t seed = 0x12345678;
    for (std::size_t i = 0; i < addresses.size(); i++) {
      seed ^= seed << 13;
      seed ^= seed >> 7;
      seed ^= seed << 17;
      const let page =
          scenario == 0   ? seed & 15
          : scenario == 1 ? (i & 1) * mem::address_space::first_level_entries
          : scenario == 2 ? (i & 1) * mem::address_space::second_level_entries
          : scenario == 3 ? (i & 1) * 16384
                          : seed & (pages - 1);
      addresses[i] = base + page * mem::page_size + ((seed >> 16) & 63) * 8;
    }
    std::uint64_t expected = 0;
    for (const let address : addresses) {
      expected += raw(address);
      require(load_new(&memory, address) == raw(address) &&
                  legacy_memory_load(legacy.get(), address) == raw(address),
              "warmup mismatch");
    }
    std::array<std::array<double, samples>, 4> timings;
    const let direct = [&] {
      return measure(
          [&](std::uint64_t address) { return load_new(&memory, address); },
          addresses, expected);
    };
    for (std::size_t sample = 0; sample < samples; sample++) {
      for (std::size_t offset = 0; offset < timings.size(); offset++) {
        const let path = (sample + scenario + offset) % timings.size();
        switch (path) {
        case 0:
          timings[path][sample] =
              callback(legacy_memory_load, legacy.get(), addresses, expected);
          break;
        case 1:
          timings[path][sample] =
              callback(load_new, &memory, addresses, expected);
          break;
        case 2:
          timings[path][sample] = direct();
          break;
        case 3:
          timings[path][sample] = measure(raw, addresses, expected);
          break;
        }
      }
    }
    for (let &path : timings)
      std::ranges::sort(path);
    constexpr std::array names{"hot_16_pages", "L1_conflicts", "large_stride",
                               "legacy_cache_conflicts", "random_4096_pages"};
    std::println("{},{:.3f},{:.3f},{:.3f},{:.3f}", names[scenario],
                 timings[0][samples / 2], timings[1][samples / 2],
                 timings[2][samples / 2], timings[3][samples / 2]);
  }
  return 0;
}
} // namespace

fn main() -> int {
  try {
    return run();
  } catch (const std::exception &error) {
    std::println(stderr, "memory benchmark: {}", error.what());
    return 1;
  }
}
