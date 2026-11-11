#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <string_view>
#include <cstdlib>
#include <exception>
#include <memory>
#include <print>
#include <span>
#include <variant>
#include <vector>
#include <unistd.h>
#include <elf.h>
#include <gelf.h>
#include <libelf.h>

#include <dynamips/macros.h>

import devices.rom_services;
import machine.firmware;
import machine.core;
import machine.c7200;
import native_fd;
import jit.memory;
import jit.uart;
import jit.devices;
import jit.dispatch;

namespace {
namespace mem = jit::memory;
namespace machine = dynamips::machine;
namespace firmware = dynamips::firmware;

fn check(bool condition, const char *message) -> bool {
  if (!condition)
    std::println(stderr, "machine test: {}", message);
  return condition;
}

struct fixture {
  char path[40] = "/tmp/dynamips-firmware-XXXXXX";
  dynamips::host::file_descriptor descriptor{mkstemp(path)};

  ~fixture() { static_cast<void>(unlink(path)); }
};

struct end_elf {
  fn operator()(Elf *value) const noexcept -> void { elf_end(value); }
};

fn write_elf(fixture &file, std::uint32_t address,
               std::span<const std::uint32_t> words, std::uint32_t bss,
               std::uint16_t architecture = EM_MIPS) -> bool {
  if (!check(file.descriptor.get() >= 0 &&
                 ftruncate(file.descriptor.get(), 0) == 0 &&
                 lseek(file.descriptor.get(), 0, SEEK_SET) == 0 &&
                 elf_version(EV_CURRENT) != EV_NONE,
             "prepare real ELF fixture"))
    return false;
  std::unique_ptr<Elf, end_elf> elf{
      elf_begin(file.descriptor.get(), ELF_C_WRITE, nullptr)};
  if (!check(elf && gelf_newehdr(elf.get(), ELFCLASS32), "create ELF32 header"))
    return false;
  GElf_Ehdr header{};
  if (!check(gelf_getehdr(elf.get(), &header) != nullptr,
             "read new ELF header"))
    return false;
  header.e_ident[EI_DATA] = ELFDATA2MSB;
  header.e_machine = architecture;
  header.e_type = ET_EXEC;
  header.e_version = EV_CURRENT;
  header.e_entry = address;
  if (!check(gelf_update_ehdr(elf.get(), &header) != 0,
             "write big endian MIPS header"))
    return false;
  std::vector<std::byte> bytes(words.size() * 4);
  for (std::size_t i = 0; i < words.size(); i++)
    for (unsigned j = 0; j < 4; j++)
      bytes[i * 4 + j] =
          static_cast<std::byte>((words[i] >> (24 - j * 8)) & 255);
  Elf_Scn *text = elf_newscn(elf.get());
  Elf_Data *data = text ? elf_newdata(text) : nullptr;
  if (!check(text && data, "create executable section"))
    return false;
  GElf_Shdr description{};
  description.sh_type = SHT_PROGBITS;
  description.sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  description.sh_addr = address;
  description.sh_addralign = 4;
  data->d_buf = bytes.data();
  data->d_size = bytes.size();
  data->d_align = 4;
  data->d_type = ELF_T_BYTE;
  data->d_version = EV_CURRENT;
  if (!check(gelf_update_shdr(text, &description) != 0,
             "write executable section metadata"))
    return false;
  Elf_Scn *zero = elf_newscn(elf.get());
  Elf_Data *zero_data = zero ? elf_newdata(zero) : nullptr;
  if (!check(zero && zero_data, "create BSS section"))
    return false;
  description = {};
  description.sh_type = SHT_NOBITS;
  description.sh_flags = SHF_ALLOC | SHF_WRITE;
  description.sh_addr = bss;
  description.sh_addralign = 4;
  zero_data->d_size = 16;
  zero_data->d_align = 4;
  zero_data->d_type = ELF_T_BYTE;
  zero_data->d_version = EV_CURRENT;
  return check(gelf_update_shdr(zero, &description) != 0 &&
                   elf_update(elf.get(), ELF_C_WRITE) > 0,
               "write real ELF fixture to disk");
}

fn rom_contract() -> bool {
  std::array<std::byte, mem::page_size> ram{}, rom{}, other{};
  let space = mem::address_space::create(ram, 4);
  if (!check(space.has_value(), "create ROM address space"))
    return false;
  let &memory = **space;
  if (!check(memory.map_ram({0x1000}, {0},
                            mem::permission::read | mem::permission::write)
                     .has_value() &&
                 !memory.map_rom({0x2000}, {0x100000}, ram) &&
                 memory.map_rom({0x2000}, {0x100000}, rom).has_value() &&
                 memory.map_rom({0x3000}, {0x100000}, rom).has_value(),
             "ROM backing cannot alias writable RAM and supports consistent "
             "ROM aliases"))
    return false;
  if (!check(!memory.map_rom({0x4000}, {0x100000}, other),
             "reject inconsistent physical ROM aliases"))
    return false;
  const let write = memory.store<std::uint32_t>({0x2000}, 1);
  const let protect =
      memory.protect({0x3000}, mem::permission::read | mem::permission::write);
  return check(
      !write && write.error().reason == mem::fault_reason::permission &&
          !protect &&
          protect.error() == mem::mapping_error::invalid_permissions &&
          memory.fetch<std::uint32_t>({0x2000}) == 0 &&
          memory.seal_executable_pages().has_value(),
      "ROM is executable but cannot gain write permission through an alias");
}

fn composition() -> bool {
  fixture rom_file, ios_file;
  constexpr std::array boot_words{0x34010007u, 0u};
  constexpr std::array ios_words{0x34020009u, 0u};
  if (!write_elf(rom_file, 0xbfc00000, boot_words, 0xa0002000) ||
      !write_elf(ios_file, 0x80003000, ios_words, 0xa0003100))
    return false;
  let rom = firmware::image::read(rom_file.path);
  let ios = firmware::image::read(ios_file.path);
  if (!check(rom && ios && rom->entry() == 0xffffffffbfc00000ULL &&
                 rom->sections().size() == 2,
             "load and sign-extend big endian firmware addresses"))
    return false;
  constexpr std::array windows{
      machine::device_window{"fixture-uart", {{0x100000}, mem::page_size}}};
  const machine::definition model{
      "fixture-mips",
      {"R7000", 0x2721, {0xffffffffbfc00000ULL}, {0xffffffff80004000ULL}},
      4 * mem::page_size,
      {{0x1fc00000}, mem::page_size},
      windows};
  jit::uart uart{{0x100000}};
  const std::array attachments{machine::attachment{
      {0},
      {&uart.endpoint(),
       jit::devices::widths::word,
       std::endian::big,
       {&uart,
        [](void *value) noexcept { static_cast<jit::uart *>(value)->reset(); },
        nullptr}}}};
  let board = machine::instance::create(model, *rom, &*ios, attachments);
  if (!check(board.has_value(),
             "compose CPU, firmware, RAM, ROM and attached UART"))
    return false;
  let &memory = (*board)->memory();
  if (!check(
          (*board)->cpu().pc.value == model.cpu.reset.value &&
              (*board)->ios_entry() == 0xffffffff80003000ULL &&
              memory.fetch<std::uint32_t>({model.cpu.reset.value}) ==
                  boot_words[0] &&
              memory.load<std::uint32_t>({machine::cached_base + 0x3000}) ==
                  ios_words[0] &&
              memory.load<std::uint32_t>({machine::uncached_base + 0x3100}) ==
                  0 &&
              memory.load<std::uint32_t>({machine::cached_base + 0x2000}) == 0,
          "reset fetch, IOS placement, BSS and RAM aliases use the model"))
    return false;
  const let first = (*board)->run_slice(100);
  if (!check(first && first->reason == jit::stop_reason::completed &&
                 (*board)->cpu().registers[1] == 7,
             "execute firmware through the shared machine owner"))
    return false;
  if (!check(
          memory.store<std::uint32_t>({machine::cached_base + 0x2000},
                                      0x12345678)
                  .has_value() &&
              memory
                  .store<std::uint32_t>({machine::uncached_base + 0x100000},
                                        'A')
                  .has_value() &&
              uart.output() == "A",
          "attach a device using data rather than chassis initialization code"))
    return false;
  (*board)->reset();
  if (!check((*board)->cpu().pc.value == model.cpu.reset.value &&
                 (*board)->cpu().registers[1] == 0 &&
                 (*board)->cpu().registers[29] == model.cpu.boot_stack.value &&
                 uart.output().empty() &&
                 memory.load<std::uint32_t>(
                     {machine::uncached_base + 0x2000}) == 0x12345678,
             "warm reset restores CPU/devices and preserves RAM"))
    return false;
  const let warm = (*board)->run_slice(100);
  if (!check(warm && warm->compiled == 0,
             "warm boot safely reuses immutable ROM code"))
    return false;
  const std::array duplicate{attachments[0], attachments[0]};
  if (!check(!machine::instance::create(model, *rom, nullptr, duplicate),
             "reject duplicate device attachments"))
    return false;
  let invalid = model;
  invalid.ram_bytes = 1;
  if (!check(!machine::instance::create(invalid, *rom),
             "reject invalid layouts before mapping"))
    return false;
  constexpr std::array large_words{0u};
  if (!write_elf(ios_file, 0x80010000, large_words, 0xa0010010))
    return false;
  let outside = firmware::image::read(ios_file.path);
  if (!check(outside && !machine::instance::create(model, *rom, &*outside),
             "reject firmware beyond owned RAM"))
    return false;
  if (!write_elf(ios_file, 0x80003000, ios_words, 0xa0003100, EM_386))
    return false;
  const let wrong = firmware::image::read(ios_file.path);
  if (!check(!wrong &&
                 wrong.error().code == firmware::error_code::unsupported_format,
             "reject another instruction architecture"))
    return false;
  if (!write_elf(ios_file, 0x80003000, ios_words, 0x80003004))
    return false;
  const let overlapping = firmware::image::read(ios_file.path);
  if (!check(!overlapping && overlapping.error().code ==
                                 firmware::error_code::invalid_section,
             "reject overlapping sections while retaining the ELF backing "
             "through cleanup"))
    return false;
  if (!check(ftruncate(ios_file.descriptor.get(), 8) == 0 &&
                 !firmware::image::read(ios_file.path),
             "reject truncated ELF input"))
    return false;
  if (!check(ftruncate(ios_file.descriptor.get(),
                       firmware::image::max_file_size + 1) == 0,
             "create sparse oversized firmware fixture"))
    return false;
  const let oversized = firmware::image::read(ios_file.path);
  if (!check(!oversized &&
                 oversized.error().code == firmware::error_code::too_large,
             "reject oversized firmware before allocating its payload"))
    return false;
  return true;
}

fn rom_service_queues() -> bool {
  using dynamips::devices::rom_output;
  dynamips::devices::rom_services device{{0x16000000}};
  jit::devices::bus bus;
  if (!check(bus.add({{0x16000000}, mem::page_size}, device.descriptor())
                 .has_value(),
             "register ROM services"))
    return false;
  std::array<std::byte, mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 1);
  if (!check(memory && (*memory)
                           ->map_device({0x8000}, {0x16000000}, bus.endpoint(),
                                        mem::permission::read |
                                            mem::permission::write)
                           .has_value(),
             "map ROM service registers"))
    return false;
  let &guest = **memory;
  if (!check(!guest.store<std::uint8_t>({0x8040}, 'R') &&
                 !guest.store<std::uint32_t>({0x8044}, 1) &&
                 !guest.load<std::uint32_t>({0x8040}),
             "reject unsupported ROM service widths, offsets and reads"))
    return false;
  if (!check(guest.store<std::uint32_t>({0x803c}, 'L').has_value(),
             "queue ROM logging separately"))
    return false;
  for (std::size_t i = 0; i < device.capacity; i++)
    if (!check(guest
                   .store<std::uint32_t>(
                       {0x8040},
                       0x12340000u | static_cast<std::uint32_t>(i & 0xff))
                   .has_value(),
               "queue console low bytes"))
      return false;
  const let blocked =
      device.drain(rom_output::console,
                   [](std::span<const std::byte>) noexcept
                       -> std::expected<std::size_t, int> { return 0; });
  if (!check(blocked && *blocked == 0 &&
                 !guest.store<std::uint32_t>({0x8040}, 'X'),
             "backpressure preserves a full console queue without overwriting"))
    return false;
  std::size_t received = 0;
  const let drained = device.drain(
      rom_output::console,
      [&](std::span<const std::byte> bytes) noexcept
          -> std::expected<std::size_t, int> {
        const std::size_t count = bytes.size() < 3 ? bytes.size() : 3;
        for (std::size_t i = 0; i < count; i++)
          if (bytes[i] != static_cast<std::byte>((received + i) & 0xff))
            return std::unexpected(1);
        received += count;
        return count;
      });
  if (!check(drained && *drained == device.capacity &&
                 received == device.capacity,
             "partial drains preserve every console byte and its order"))
    return false;
  const let log =
      device.drain(rom_output::log,
                   [](std::span<const std::byte> bytes) noexcept
                       -> std::expected<std::size_t, int> {
                     if (bytes.size() != 1 || bytes[0] != std::byte{'L'})
                       return std::unexpected(1);
                     return bytes.size();
                   });
  if (!check(log && *log == 1 &&
                 guest.store<std::uint32_t>({0x8040}, 'R').has_value() &&
                 guest.store<std::uint32_t>({0x803c}, 'L').has_value(),
             "console and log queues are independent and reusable"))
    return false;
  bus.reset();
  for (const let channel : {rom_output::console, rom_output::log}) {
    const let empty = device.drain(
        channel,
        [](std::span<const std::byte> bytes) noexcept
            -> std::expected<std::size_t, int> { return bytes.size(); });
    if (!check(empty && *empty == 0, "reset clears pending ROM output"))
      return false;
  }
  return true;
}

fn real_rom(const char *path) -> bool {
  let rom_image = firmware::image::read(path);
  if (!check(rom_image.has_value(), "load bundled real MIPS ROM"))
    return false;
  using dynamips::devices::rom_output;
  dynamips::devices::rom_services services{{0x16000000}};
  const std::array attachments{machine::attachment{
      dynamips::c7200::rom_services, services.descriptor()}};
  let board = machine::instance::create(dynamips::c7200::npe400, *rom_image,
                                         nullptr, attachments);
  if (!check(board.has_value(), "prepare full-size C7200 NPE-400 memory map"))
    return false;
  const let first =
      (*board)->memory().fetch<std::uint32_t>({0xffffffffbfc00000ULL});
  std::array<std::array<char, 512>, 2> output{};
  std::array<std::size_t, 2> written{};
  bool output_ok = true;
  std::uint64_t instructions = 0;
  const let boot = [&] {
    for (;;) {
      let result = (*board)->run_slice(4096 - instructions);
      for (std::size_t i = 0; i < output.size(); i++) {
        const let channel = i == 0 ? rom_output::console : rom_output::log;
        const let drained =
            services.drain(channel,
                           [&](std::span<const std::byte> bytes) noexcept
                               -> std::expected<std::size_t, int> {
                             if (bytes.size() > output[i].size() - written[i])
                               return std::unexpected(1);
                             std::memcpy(output[i].data() + written[i],
                                         bytes.data(), bytes.size());
                             written[i] += bytes.size();
                             return bytes.size();
                           });
        output_ok = output_ok && drained.has_value();
      }
      if (!result || result->reason != jit::stop_reason::yielded)
        return result;
      instructions += result->instructions;
    }
  }();
  if (!check(
          output_ok && first == 0x0ff004a8 && !boot &&
              std::holds_alternative<jit::dispatch_error>(boot.error().cause),
          "real C7200 ROM reaches the current CPU implementation boundary"))
    return false;
  const let &error = std::get<jit::dispatch_error>(boot.error().cause);
  return check(
      error.code == jit::dispatch_error_code::translation &&
          error.pc == 0xffffffffbfc012e0ULL &&
          (*board)->memory().fetch<std::uint32_t>({error.pc}) == 0x50620009 &&
          (*board)->cpu().registers[29] == 0xffffffff80003fc8ULL &&
          (*board)->memory().load<std::uint64_t>({0xffffffff80003ff8ULL}) ==
              0xffffffffbfc00008ULL &&
          std::string_view{output[0].data(), written[0]} ==
              "ROMMON emulation microcode.\n\n" &&
          std::string_view{output[1].data(), written[1]} ==
              "Microcode has started.\n",
      "real ROM prints both startup messages before the next CPU boundary");
}
} // namespace

fn main(int argc, char **argv) -> int {
  try {
    if (argc != 2)
      return 1;
    if (!rom_contract() || !composition() || !rom_service_queues() ||
        !real_rom(argv[1]))
      return 1;
    std::println("machine: ELF loading, chassis composition, ROM protection, "
                 "reset and real C7200 boot boundary passed");
    return 0;
  } catch (const std::exception &error) {
    std::println(stderr, "machine test: {}", error.what());
    return 1;
  }
}
