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
#include <minizip/zip.h>

#include <dynamips/macros.h>

import devices.byte_memory;
import devices.c7200_midplane;
import devices.gt64120;
import devices.c7200_iofpga;
import devices.rom_services;
import machine.firmware;
import machine.core;
import machine.c7200;
import host_memory;
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

struct zip_fixture {
  char path[48] = "/tmp/dynamips-archive-XXXXXX.zip";
  dynamips::host::file_descriptor descriptor{mkstemps(path, 4)};

  ~zip_fixture() { static_cast<void>(unlink(path)); }
};

fn write_zip(const char *path, std::span<const char *const> names,
               std::span<const std::byte> bytes, int method) -> bool {
  zipFile archive = zipOpen64(path, APPEND_STATUS_CREATE);
  if (!check(archive != nullptr, "create ZIP fixture"))
    return false;
  bool good = true;
  for (const char *name : names) {
    if (zipOpenNewFileInZip64(archive, name, nullptr, nullptr, 0, nullptr, 0,
                              nullptr, method, Z_DEFAULT_COMPRESSION,
                              1) != ZIP_OK) {
      good = false;
      break;
    }
    const int written = zipWriteInFileInZip(
        archive, bytes.data(), static_cast<unsigned>(bytes.size()));
    const int closed = zipCloseFileInZip(archive);
    if (written != ZIP_OK || closed != ZIP_OK) {
      good = false;
      break;
    }
  }
  const int closed = zipClose(archive, nullptr);
  return check(good && closed == ZIP_OK, "write and close ZIP fixture");
}

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

fn controller_configuration() -> bool {
  dynamips::devices::gt64120 controller{{0x14000000}};
  jit::devices::bus bus;
  if (!check(bus.add({{0x14000000}, mem::page_size}, controller.descriptor())
                 .has_value(),
             "attach system controller"))
    return false;
  std::array<std::byte, mem::page_size> ram{};
  let memory = mem::address_space::create(ram, 1);
  if (!check(memory && (*memory)
                           ->map_device({0x4000}, {0x14000000}, bus.endpoint(),
                                        mem::permission::read |
                                            mem::permission::write)
                           .has_value(),
             "map little-endian controller into big-endian guest"))
    return false;
  const let write_address = [&](std::uint64_t offset, std::uint32_t value) {
    return (*memory)->store<std::uint32_t>({0x4000 + offset},
                                           std::byteswap(value));
  };
  if (!check((*memory)->load<std::uint32_t>({0x4c00}) == 0x01800000 &&
                 write_address(0x000, 0xffffffff).has_value() &&
                 (*memory)->load<std::uint32_t>({0x4000}) == 0 &&
                 write_address(0x864, 0).has_value() &&
                 (*memory)->load<std::uint32_t>({0x4864}) == 0 &&
                 !write_address(0x864, 1) &&
                 (*memory)->load<std::uint32_t>({0x4020}) == 0x7f000000 &&
                 write_address(0xcf8, 0x80000000).has_value() &&
                 (*memory)->load<std::uint32_t>({0x4cfc}) == 0xab112046 &&
                 write_address(0xcf8, 0x80000008).has_value() &&
                 (*memory)->load<std::uint32_t>({0x4cfc}) == 0x05800003,
             "GT64120 startup and PCI identity registers preserve legacy byte "
             "order"))
    return false;
  if (!check(write_address(0xcf8, 0x80000800).has_value() &&
                 (*memory)->load<std::uint32_t>({0x4cfc}) == 0xffffffff &&
                 write_address(0xcf0, 0x80000000).has_value() &&
                 (*memory)->load<std::uint32_t>({0x4cf4}) == 0xffffffff &&
                 (*memory)->load<std::uint32_t>({0x4cf8}) ==
                     std::byteswap(0x80000800u),
             "empty PCI slots return absent IDs with independent configuration "
             "addresses"))
    return false;
  if (!check(
          write_address(0xc1c, 0x12345678).has_value() &&
              (*memory)->load<std::uint32_t>({0x4c1c}) == 0x78563412 &&
              (*memory)->load<std::uint32_t>({0x4c18}) == 0 &&
              !(*memory)->load<std::uint32_t>({0x4840}),
          "interrupt mask retains state while unsupported DMA registers fault"))
    return false;
  for (const std::uint64_t offset :
       {0x48ULL,  0x50ULL,  0x58ULL,  0x60ULL,  0x80ULL,  0x88ULL,  0x90ULL,
        0x98ULL,  0xa0ULL,  0xa8ULL,  0xb0ULL,  0xb8ULL,  0xc04ULL, 0xc10ULL,
        0xc14ULL, 0xc24ULL, 0xc3cULL, 0xc40ULL, 0xc48ULL, 0xc58ULL, 0xc80ULL,
        0xc84ULL, 0xc88ULL, 0xc8cULL, 0xc90ULL, 0xc94ULL, 0xca4ULL, 0xcbcULL,
        0xcc0ULL, 0xcc8ULL, 0xcd8ULL}) {
    if (!check(
            write_address(offset, 0x12345678).has_value() &&
                (*memory)->load<std::uint32_t>({0x4000 + offset}) == 0,
            "controller decode-window setup matches legacy inactive windows"))
      return false;
  }
  using dynamips::devices::bridge_model;
  using dynamips::devices::pci_bridge_definition;
  const std::array invalid{pci_bridge_definition{2, 1, bridge_model::dec21050}};
  const std::array duplicate{
      pci_bridge_definition{0, 1, bridge_model::dec21050},
      pci_bridge_definition{0, 1, bridge_model::dec21150}};
  if (!check(!controller.configure_bridges(invalid) &&
                 !controller.configure_bridges(duplicate) &&
                 controller.configure_bridges(dynamips::c7200::npe400_bridges)
                     .has_value(),
             "validate and install NPE-400 bridge definitions"))
    return false;
  for (const let &bridge : dynamips::c7200::npe400_bridges) {
    const std::uint64_t selector = bridge.root ? 0xcf0 : 0xcf8;
    const std::uint64_t data = 0x4000 + selector + 4;
    const std::uint32_t address =
        0x80000000u | (std::uint32_t{bridge.device} << 11);
    const std::uint32_t identity =
        (static_cast<std::uint32_t>(bridge.model) << 16) | 0x1011u;
    if (!check(write_address(selector, address).has_value() &&
                   (*memory)->load<std::uint32_t>({data}) ==
                       std::byteswap(identity) &&
                   write_address(selector, address | 8).has_value() &&
                   (*memory)->load<std::uint32_t>({data}) ==
                       std::byteswap(0x06040000u) &&
                   write_address(selector, address | 0x18).has_value() &&
                   (*memory)
                       ->store<std::uint32_t>(
                           {data}, std::byteswap(0x00030200u + bridge.device))
                       .has_value() &&
                   (*memory)->load<std::uint32_t>({data}) ==
                       std::byteswap(0x00030200u + bridge.device),
               "built-in PCI bridges expose identities, class and programmable "
               "bus numbers"))
      return false;
  }
  if (!check(write_address(0xcf8, 0x80000818).has_value() &&
                 (*memory)->load<std::uint32_t>({0x4cfc}) ==
                     std::byteswap(0x00030201u),
             "bridge configuration is independent across devices and roots"))
    return false;
  bus.reset();
  if (!check(write_address(0xcf8, 0x80000818).has_value() &&
                 (*memory)->load<std::uint32_t>({0x4cfc}) == 0,
             "reset clears bridge bus-number programming"))
    return false;
  if (!write_address(0xcf8, 0))
    return false;
  return check((*memory)->load<std::uint32_t>({0x4c1c}) == 0 &&
                   (*memory)->load<std::uint32_t>({0x4cf8}) == 0,
               "controller reset restores masks and configuration selectors");
}

fn empty_port_adapters() -> bool {
  dynamips::devices::c7200_midplane midplane;
  let &endpoint = *midplane.descriptor().endpoint;
  const let access = [](std::uint64_t offset) {
    return mem::device_access{
        {0x1e800000 + offset}, std::endian::big, mem::access_size::word};
  };
  if (!check(endpoint.write(endpoint.context, access(0x60), 0x02020202)
                     .has_value() &&
                 endpoint.read(endpoint.context, access(0x60)) == 0x42424242 &&
                 endpoint.read(endpoint.context, access(0x68)) == 3 &&
                 endpoint.write(endpoint.context, access(0x68), 0x203)
                     .has_value() &&
                 endpoint.read(endpoint.context, access(0x68)) == 0x4203 &&
                 endpoint.read(endpoint.context, access(0x50)) == 6,
             "empty adapter bays expose pulled-high EEPROM lines and I/O-slot "
             "power"))
    return false;
  if (!check(endpoint.write(endpoint.context, access(0x20), 0xffffffff)
                     .has_value() &&
                 endpoint.read(endpoint.context, access(0x20)) == 0xffffffff &&
                 endpoint.read(endpoint.context, access(0x10)) == 0 &&
                 !endpoint.write(endpoint.context, access(0x68), 0),
             "adapter masks retain state without fabricating interrupts; "
             "unsupported shutdown faults"))
    return false;
  midplane.reset();
  return check(endpoint.read(endpoint.context, access(0x60)) == 0 &&
                   endpoint.read(endpoint.context, access(0x68)) == 3 &&
                   endpoint.read(endpoint.context, access(0x20)) == 0,
               "midplane reset restores EEPROM pins and interrupt masks");
}

fn serial_board_identity() -> bool {
  dynamips::devices::c7200_iofpga io{dynamips::c7200::npe400_eeprom,
                                     dynamips::c7200::vxr_eeprom};
  let &endpoint = *io.descriptor().endpoint;
  const mem::device_access pins{
      {0x1e84021c}, std::endian::big, mem::access_size::word};
  const let read_word =
      [&](unsigned chip,
          unsigned address) -> std::expected<std::uint16_t, mem::device_error> {
    const unsigned select = 4u << (chip * 3);
    const unsigned clock = 2u << (chip * 3);
    const unsigned data = 1u << (chip * 3);
    const unsigned output = 64u << chip;
    if (let result = endpoint.write(endpoint.context, pins, 0); !result)
      return std::unexpected(result.error());
    if (let result = endpoint.write(endpoint.context, pins, select); !result)
      return std::unexpected(result.error());
    // Leading zero clocks do not begin a command.
    if (let result = endpoint.write(endpoint.context, pins, select | clock);
        !result)
      return std::unexpected(result.error());
    const unsigned command = 3 | (address << 3);
    for (unsigned bit = 0; bit < 9; bit++) {
      const unsigned value = select | ((command & (1u << bit)) ? data : 0u);
      if (let result = endpoint.write(endpoint.context, pins, value); !result)
        return std::unexpected(result.error());
      if (let result = endpoint.write(endpoint.context, pins, value | clock);
          !result)
        return std::unexpected(result.error());
    }
    std::uint16_t word = 0;
    for (unsigned bit = 0; bit < 16; bit++) {
      if (let result = endpoint.write(endpoint.context, pins, select); !result)
        return std::unexpected(result.error());
      if (let result = endpoint.write(endpoint.context, pins, select | clock);
          !result)
        return std::unexpected(result.error());
      const let value = endpoint.read(endpoint.context, pins);
      if (!value)
        return std::unexpected(value.error());
      if (*value & output)
        word |= static_cast<std::uint16_t>(1u << bit);
    }
    return word;
  };
  if (!check(read_word(0, 0) == 0x04ff && read_word(0, 26) == 0x3430 &&
                 read_word(0, 63) == 0xffff && read_word(1, 0) == 0x0106 &&
                 read_word(1, 1) == 0x0201 && read_word(1, 63) == 0xffff,
             "clocked CPU/VXR EEPROM reads select independent chips and bound "
             "addresses"))
    return false;
  io.reset();
  return check(read_word(0, 1) == 0x4001 && read_word(1, 1) == 0x0201,
               "reset clears serial command state and preserves board images");
}

fn io_console() -> bool {
  dynamips::devices::c7200_iofpga io;
  let &strobe = *io.descriptor().endpoint;
  const mem::device_access startup{
      {0x1e840398}, std::endian::big, mem::access_size::word};
  if (!check(
          strobe.write(strobe.context, startup, 0x02000000).has_value() &&
              strobe.read(strobe.context, startup) == 0 &&
              strobe.write(strobe.context, startup, 0).has_value(),
          "I/O FPGA startup strobes are accepted and read zero as in legacy"))
    return false;

  let &endpoint = *io.descriptor().endpoint;
  const let access = [](std::uint64_t offset) {
    return mem::device_access{
        {0x1e840000 + offset}, std::endian::big, mem::access_size::byte};
  };
  if (!endpoint.write(endpoint.context, access(0x414), 5) ||
      !endpoint.write(endpoint.context, access(0x454), 4))
    return false;
  const std::array input{std::byte{'C'}};
  if (!check(endpoint.write(endpoint.context, access(0x41c), 'A').has_value() &&
                 endpoint.write(endpoint.context, access(0x45c), 'B')
                     .has_value() &&
                 io.receive(input) == 1 &&
                 endpoint.read(endpoint.context, access(0x40c)) == 5 &&
                 endpoint.read(endpoint.context, access(0x41c)) == 'C' &&
                 endpoint.read(endpoint.context, access(0x40c)) == 4,
             "SCN2681 status and RX follow queued console data"))
    return false;
  bool correct = true;
  for (const bool auxiliary : {false, true}) {
    const let drained = io.drain(
        auxiliary,
        [&](std::span<const std::byte> bytes) noexcept
            -> std::expected<std::size_t, int> {
          correct = correct && bytes.size() == 1 &&
                    bytes[0] == (auxiliary ? std::byte{'B'} : std::byte{'A'});
          return bytes.size();
        });
    if (!check(drained && *drained == 1 && correct,
               "console and auxiliary TX queues remain separate"))
      return false;
  }
  for (unsigned i = 0; i < jit::uart::capacity; i++)
    if (!endpoint.write(endpoint.context, access(0x41c), 'X'))
      return false;
  if (!check(endpoint.read(endpoint.context, access(0x40c)) == 0 &&
                 !endpoint.write(endpoint.context, access(0x41c), 'Y'),
             "full console queue removes TX-ready and rejects overflow"))
    return false;
  bool first = true;
  const let partial = io.drain(false,
                               [&](std::span<const std::byte>) noexcept
                                   -> std::expected<std::size_t, int> {
                                 if (!first)
                                   return 0;
                                 first = false;
                                 return 17;
                               });
  if (!check(partial && *partial == 17 &&
                 endpoint.read(endpoint.context, access(0x40c)) == 4,
             "partial host drain restores TX readiness"))
    return false;
  io.reset();
  const let empty =
      io.drain(false,
               [](std::span<const std::byte> bytes) noexcept
                   -> std::expected<std::size_t, int> { return bytes.size(); });
  return check(
      empty && *empty == 0 &&
          endpoint.read(endpoint.context, access(0x204)) == 4,
      "I/O FPGA reset clears queues and restores packed NVRAM control");
}

fn mapped_memory() -> bool {
  fixture rom_file;
  constexpr std::array words{0x3c016000u, // lui r1, 0x6000
                             0x40815000u, // mtc0 r1, EntryHi
                             0x34010006u, // ori r1, r0, V|D
                             0x40811000u, // mtc0 r1, EntryLo0
                             0x34010046u, // physical page 0x1000, V|D
                             0x40811800u, // mtc0 r1, EntryLo1
                             0x42000002u, // tlbwi
                             0x3c026000u, 0x34031234u,
                             0xac430000u, // sw r3, 0(r2)
                             0x8c440000u, // lw r4, 0(r2)
                             0x8c451000u, // lw r5, 0x1000(r2)
                             0x1000ffffu, 0u};
  if (!write_elf(rom_file, 0xbfc00000, words, 0xbfc00100))
    return false;
  let rom = firmware::image::read(rom_file.path);
  if (!check(rom.has_value(), "load mapped-memory firmware"))
    return false;
  const machine::definition model{
      "fixture-tlb",
      {"R7000", 0x2721, {0xffffffffbfc00000ULL}, {}},
      4 * mem::page_size,
      {{0x1fc00000}, mem::page_size},
      {}};
  let board = machine::instance::create(model, *rom, nullptr, {},
                                         machine::boot_mode::run);
  if (!check(board.has_value(), "create mapped-memory machine"))
    return false;
  let &memory = (*board)->memory();
  if (!check(
          memory.store<std::uint32_t>({machine::cached_base + 0x1000}, 0x5678)
              .has_value(),
          "seed odd physical page"))
    return false;
  const let run = (*board)->run_slice(40);
  if (!check(run && (*board)->cpu().registers[4] == 0x1234 &&
                 (*board)->cpu().registers[5] == 0x5678 &&
                 memory.load<std::uint32_t>({machine::cached_base}) == 0x1234,
             "guest TLBWI routes paired mapped pages to physical RAM"))
    return false;
  (*board)->reset();
  return check(!memory.load<std::uint32_t>({0x60000000}),
               "reset discards cached TLB translations");
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
  if (!check(rom && ios && ios->machine_id() == 8 &&
                 rom->entry() == 0xffffffffbfc00000ULL &&
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
  let board = machine::instance::create(model, *rom, &*ios, attachments,
                                         machine::boot_mode::handoff);
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
  if (!check(memory.load_linked<std::uint32_t>({machine::cached_base + 0x2000})
                 .has_value(),
             "reserve RAM before reset"))
    return false;
  (*board)->reset();
  if (!check(memory.store_conditional<std::uint32_t>(
                 {machine::cached_base + 0x2000}, 0) == false,
             "reset clears reservation"))
    return false;
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

fn archive_contract() -> bool {
  fixture elf_file;
  zip_fixture zip_file;
  if (!check(zip_file.descriptor.get() >= 0, "create temporary archive file"))
    return false;
  constexpr std::array words{0x34010007u, 0u};
  if (!write_elf(elf_file, 0x80008000, words, 0x80009000))
    return false;
  let source = dynamips::host::read_file(elf_file.path);
  if (!check(source.has_value(), "read ELF for ZIP fixtures"))
    return false;
  const let bytes = std::as_bytes(std::span{source->bytes.get(), source->size});
  constexpr std::array names{"nested/IOS.BIN"};
  for (const int method : {0, Z_DEFLATED}) {
    if (!write_zip(zip_file.path, names, bytes, method))
      return false;
    let loaded = firmware::image::read(zip_file.path);
    if (!check(loaded && loaded->entry() == 0xffffffff80008000ULL &&
                   loaded->sections().size() == 2,
               "load stored and deflated ZIP64 members with case-insensitive "
               "extensions"))
      return false;
    let moved = std::move(*loaded);
    if (!check(moved.sections()[0].contents[0] == std::byte{0x34},
               "archive section views retain owned decompressed storage after "
               "moving"))
      return false;
  }
  constexpr std::array multiple{"one.bin", "two.image"};
  if (!write_zip(zip_file.path, multiple, bytes, 0))
    return false;
  const let ambiguous = firmware::image::read(zip_file.path);
  if (!check(!ambiguous && ambiguous.error().code ==
                               firmware::error_code::multiple_archive_images,
             "reject multiple IOS members"))
    return false;
  constexpr std::array missing{"readme.txt"};
  if (!write_zip(zip_file.path, missing, bytes, 0))
    return false;
  const let absent = firmware::image::read(zip_file.path);
  if (!check(!absent && absent.error().code ==
                            firmware::error_code::missing_archive_image,
             "reject archives without an IOS member"))
    return false;
  if (!write_zip(zip_file.path, names, {}, 0))
    return false;
  const let empty = firmware::image::read(zip_file.path);
  if (!check(!empty && empty.error().code ==
                           firmware::error_code::missing_archive_image,
             "reject empty IOS members"))
    return false;
  if (!write_zip(zip_file.path, names, bytes, 0))
    return false;
  let archive = dynamips::host::read_file(zip_file.path);
  if (!check(archive && archive->size >= 30, "read stored ZIP metadata"))
    return false;
  const let *header = archive->bytes.get();
  const std::size_t data_offset = 30 + header[26] +
                                  (std::size_t{header[27]} << 8) + header[28] +
                                  (std::size_t{header[29]} << 8);
  const unsigned char corrupt = 0;
  if (!check(data_offset < archive->size &&
                 pwrite(zip_file.descriptor.get(), &corrupt, 1,
                        static_cast<off_t>(data_offset)) == 1,
             "corrupt ZIP payload without updating CRC"))
    return false;
  const let crc = firmware::image::read(zip_file.path);
  if (!check(!crc && crc.error().code == firmware::error_code::invalid_archive,
             "verify member CRC before parsing ELF"))
    return false;
  std::size_t central = 0;
  for (std::size_t i = 0; i + 46 <= archive->size; i++) {
    if (header[i] == 'P' && header[i + 1] == 'K' && header[i + 2] == 1 &&
        header[i + 3] == 2) {
      central = i;
      break;
    }
  }
  const std::array<unsigned char, 4> oversized{1, 0, 0, 16};
  if (!check(central != 0 && pwrite(zip_file.descriptor.get(), oversized.data(),
                                    oversized.size(),
                                    static_cast<off_t>(central + 24)) == 4,
             "declare an oversized ZIP member without allocating its payload"))
    return false;
  const let limited = firmware::image::read(zip_file.path);
  if (!check(!limited &&
                 limited.error().code == firmware::error_code::too_large,
             "reject oversized ZIP members before decompression"))
    return false;
  if (!check(ftruncate(zip_file.descriptor.get(), 8) == 0,
             "truncate ZIP fixture"))
    return false;
  return check(!firmware::image::read(zip_file.path),
               "reject truncated ZIP archives");
}

fn byte_memory_contract() -> bool {
  std::array<std::byte, 16> bytes{};
  dynamips::devices::byte_memory device{{0x100000}, bytes};
  let descriptor = device.descriptor();
  let &endpoint = *descriptor.endpoint;
  const mem::device_access word{
      {0x100004}, std::endian::big, mem::access_size::word};
  if (!check(endpoint.write(endpoint.context, word, 0x12345678).has_value() &&
                 endpoint.read(endpoint.context, word) == 0x12345678 &&
                 bytes[4] == std::byte{0x12} && bytes[7] == std::byte{0x78},
             "byte memory stores big-endian words"))
    return false;
  const mem::device_access little{
      {0x100004}, std::endian::little, mem::access_size::word};
  if (!check(endpoint.read(endpoint.context, little) == 0x78563412 &&
                 !endpoint.write(
                     endpoint.context,
                     {{0x100010}, std::endian::big, mem::access_size::byte}, 1),
             "byte memory respects access order and backing bounds"))
    return false;
  jit::devices::bus bus;
  if (!check(bus.add({{0x100000}, bytes.size()}, descriptor).has_value(),
             "register byte memory"))
    return false;
  bus.reset();
  return check(bytes[4] == std::byte{0x12}, "warm reset retains NVRAM bytes");
}

fn rom_service_queues() -> bool {
  dynamips::devices::rom_services geometry{{0x16000000},
                                           {.nvram = {},
                                            .rom_megabytes = 4,
                                            .nvram_kibibytes = 128,
                                            .iomem_megabytes = 0}};
  let &sizes = *geometry.descriptor().endpoint;
  if (!check(
          sizes.read(
              sizes.context,
              {{0x16000018}, std::endian::big, mem::access_size::word}) == 4 &&
              sizes.read(sizes.context, {{0x1600001c},
                                         std::endian::big,
                                         mem::access_size::word}) == 128 &&
              sizes.read(sizes.context, {{0x16000020},
                                         std::endian::big,
                                         mem::access_size::word}) == 0 &&
              sizes.read(sizes.context, {{0x1600001c},
                                         std::endian::little,
                                         mem::access_size::word}) == 0x80000000,
          "ROM memory geometry uses megabytes for ROM/IOMEM and KiB for NVRAM"))
    return false;

  const std::array environment{
      dynamips::devices::rommon_variable{"BOOT_TEST", "flash:image.bin"},
      dynamips::devices::rommon_variable{"EMPTY", ""}};
  using dynamips::devices::rom_output;
  dynamips::devices::rom_services device{{0x16000000},
                                         {{0xffffffffbe000000ULL},
                                          {0xffffffff80008000ULL},
                                          8,
                                          256,
                                          0x2142,
                                          environment}};
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
  if (!check(guest.load<std::uint32_t>({0x8000}) == 0x1e94b3df &&
                 guest.load<std::uint32_t>({0x8044}) == 0xbe000000 &&
                 guest.load<std::uint32_t>({0x8028}) == 0x80008000 &&
                 guest.load<std::uint32_t>({0x802c}) == 8 &&
                 guest.load<std::uint32_t>({0x8014}) == 256 &&
                 guest.load<std::uint32_t>({0x8024}) == 0x2142 &&
                 guest.load<std::uint32_t, std::endian::little>({0x8014}) ==
                     0x00010000 &&
                 guest.load<std::uint32_t, std::endian::little>({0x802c}) ==
                     0x08000000,
             "ROM identification register matches the real firmware"))
    return false;
  const let query = [&](std::string_view name) {
    if (!guest.store<std::uint32_t>({0x8058}, 3))
      return false;
    for (const char byte : name)
      if (!guest.store<std::uint32_t>({0x8054},
                                      static_cast<unsigned char>(byte)))
        return false;
    return guest.store<std::uint32_t>({0x8058}, 2).has_value();
  };
  const let set = [&](std::string_view assignment) {
    if (!guest.store<std::uint32_t>({0x8058}, 3))
      return false;
    for (const char byte : assignment)
      if (!guest.store<std::uint32_t>({0x8054},
                                      static_cast<unsigned char>(byte)))
        return false;
    return guest.store<std::uint32_t>({0x8058}, 1).has_value() &&
           guest.load<std::uint32_t>({0x8058}) == 0;
  };
  if (!check(query("MISSING") &&
                 guest.load<std::uint32_t>({0x8058}) == UINT32_MAX &&
                 guest.load<std::uint32_t>({0x8054}) == 0,
             "missing ROMMON variables report failure without stale data"))
    return false;
  if (!check(query("BOOT_TEST") && guest.load<std::uint32_t>({0x8058}) == 0,
             "query a borrowed boot variable"))
    return false;
  for (const char byte : std::string_view{"flash:image.bin"})
    if (!check(guest.load<std::uint32_t>({0x8054}) ==
                   static_cast<unsigned char>(byte),
               "read ROMMON value through the data register"))
      return false;
  if (!check(guest.load<std::uint32_t>({0x8054}) == 0 &&
                 guest.load<std::uint32_t>({0x8054}) == 0 && query("EMPTY") &&
                 guest.load<std::uint32_t>({0x8058}) == 0 &&
                 guest.load<std::uint32_t>({0x8054}) == 0,
             "empty ROMMON values remain distinct from missing variables"))
    return false;
  if (!check(
          set("BOOT_TEST=override") && query("BOOT_TEST") &&
              environment[0].value == "flash:image.bin",
          "guest updates shadow immutable boot defaults without changing them"))
    return false;
  for (const char byte : std::string_view{"override"})
    if (!check(guest.load<std::uint32_t>({0x8054}) ==
                   static_cast<unsigned char>(byte),
               "read updated boot variable"))
      return false;
  if (!guest.store<std::uint32_t>({0x8058}, 3))
    return false;
  for (std::size_t i = 0; i < 1024; i++)
    if (!guest.store<std::uint32_t>({0x8054}, 'X'))
      return false;
  if (!check(
          guest.store<std::uint32_t>({0x8058}, 2).has_value() &&
              guest.load<std::uint32_t>({0x8058}) == UINT32_MAX &&
              query("BOOT_TEST") && guest.load<std::uint32_t>({0x8058}) == 0 &&
              guest.store<std::uint32_t>({0x8058}, 1).has_value() &&
              guest.load<std::uint32_t>({0x8058}) == UINT32_MAX,
          "oversized queries recover and malformed assignments report failure"))
    return false;
  bus.reset();
  if (!check(guest.load<std::uint32_t>({0x8058}) == 0 &&
                 guest.load<std::uint32_t>({0x8054}) == 0 &&
                 query("BOOT_TEST") && guest.load<std::uint32_t>({0x8058}) == 0,
             "reset clears transaction state and retains the supplied "
             "environment"))
    return false;
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
  fixture ios_file;
  constexpr std::array ios_words{0x3414c0deu, 0u};
  if (!write_elf(ios_file, 0x80008000, ios_words, 0x80009000))
    return false;
  let ios = firmware::image::read(ios_file.path);
  if (!check(ios.has_value(), "load handoff target ELF"))
    return false;
  using dynamips::devices::rom_output;
  dynamips::devices::rom_services services{
      {0x16000000},
      {{0xffffffffbe000000ULL},
       {ios->entry()},
       ios->machine_id(),
       static_cast<std::uint32_t>(dynamips::c7200::npe400.ram_bytes /
                                  (1024 * 1024)),
       dynamips::c7200::configuration_register}};
  std::array<std::byte, 128 * 1024> nvram_bytes{};
  dynamips::devices::byte_memory nvram{{0x1e000000}, nvram_bytes};
  const std::array attachments{
      machine::attachment{dynamips::c7200::rom_services, services.descriptor()},
      machine::attachment{dynamips::c7200::nvram, nvram.descriptor()}};
  let board =
      machine::instance::create(dynamips::c7200::npe400, *rom_image, &*ios,
                                attachments, machine::boot_mode::handoff);
  if (!check(board.has_value(), "prepare full-size C7200 NPE-400 memory map"))
    return false;
  if (!check((*board)->cpu().cop0.registers[7] == 0x20000000,
             "NPE-400 reset exposes the RM7000 64-entry TLB feature"))
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
      if (!result || (result->reason != jit::stop_reason::yielded &&
                      result->reason != jit::stop_reason::guest_exception))
        return result;
      instructions += result->instructions;
    }
  }();
  if (!check(output_ok && first == 0x0ff004a8 && boot &&
                 boot->reason == jit::stop_reason::handoff,
             "real C7200 ROM reaches the loaded ELF handoff boundary")) {
    if (!boot &&
        std::holds_alternative<jit::dispatch_error>(boot.error().cause))
      std::println(stderr, "ROM stopped at {:#x}",
                   std::get<jit::dispatch_error>(boot.error().cause).pc);
    return false;
  }
  const bool transferred =
      check((*board)->cpu().pc.value == ios->entry() &&
                (*board)->cpu().registers[20] == 0 &&
                (*board)->cpu().registers[4] == 2 &&
                (*board)->cpu().registers[5] == 0 &&
                (*board)->cpu().registers[6] == 0 &&
                (*board)->cpu().registers[8] == ios->entry() &&
                (*board)->cpu().registers[26] == 0xffffffffbfc003e0ULL &&
                (*board)->cpu().registers[31] == 0xffffffffbfc0068cULL &&
                (*board)->cpu().registers[29] == ios->entry() - 64 &&
                (*board)->cpu().cop0.registers[9] > 0 &&
                (*board)->cpu().cop0.registers[9] < 1000000 &&
                (*board)->cpu().cop0.registers[11] == 1000000 &&
                (*board)->memory().load<std::uint64_t>(
                    {0xffffffff80003ff8ULL}) == 0xffffffffbfc00008ULL &&
                std::string_view{output[0].data(), written[0]} ==
                    "ROMMON emulation microcode.\n\nLaunching IOS image at "
                    "0x80008000...\n" &&
                std::string_view{output[1].data(), written[1]} ==
                    "Microcode has started.\n",
            "ROM transfers to the ELF entry with boot arguments and "
            "callback state intact");
  if (!transferred)
    return false;
  const let nvram_first =
      (*board)->memory().load<std::uint8_t>({0xffffffffbe000080ULL});
  (*board)->reset();
  if (!check(
          (*board)->cpu().pc.value == dynamips::c7200::npe400.cpu.reset.value &&
              (*board)->cpu().cop0.registers[12] == 0x00400000 &&
              (*board)->cpu().cop0.registers[15] == 0x2721 &&
              (*board)->cpu().cop0.registers[11] == 0 && nvram_first &&
              (*board)->memory().load<std::uint8_t>({0xffffffffbe000080ULL}) ==
                  *nvram_first,
          "reset restores CP0 bootstrap state and preserves initialized NVRAM"))
    return false;
  board->reset();
  board = machine::instance::create(dynamips::c7200::npe400, *rom_image, &*ios,
                                    attachments);
  if (!check(board.has_value(), "create RAM execution machine"))
    return false;
  std::uint64_t remaining = 4096;
  for (;;) {
    const let running = (*board)->run_slice(remaining);
    if (!check(running.has_value(), "execute the ROM and loaded RAM entry"))
      return false;
    remaining -= running->instructions;
    if (running->reason != jit::stop_reason::yielded &&
        running->reason != jit::stop_reason::guest_exception)
      return check(
          running->reason == jit::stop_reason::budget_exhausted &&
              (*board)->cpu().registers[20] == 0xc0de &&
              (*board)->cpu().pc.value > ios->entry(),
          "normal boot executes IOS instructions beyond the old handoff stop");
  }
}
} // namespace

fn main(int argc, char **argv) -> int {
  dynamips::devices::c7200_midplane midplane;
  let &endpoint = *midplane.descriptor().endpoint;
  if (!check(
          endpoint.read(
              endpoint.context,
              {{0x1e80007b}, std::endian::big, mem::access_size::byte}) == 0 &&
              endpoint
                  .write(
                      endpoint.context,
                      {{0x1e80007b}, std::endian::big, mem::access_size::byte},
                      0xff)
                  .has_value() &&
              endpoint.read(endpoint.context, {{0x1e80007b},
                                               std::endian::big,
                                               mem::access_size::byte}) == 0 &&
              !endpoint.read(
                  endpoint.context,
                  {{0x1e80007c}, std::endian::big, mem::access_size::byte}),
          "midplane priority writes are ignored and unknown registers fault"))
    return 1;
  dynamips::devices::c7200_iofpga io;
  let &probe = *io.descriptor().endpoint;
  if (!check(probe.read(probe.context, {{0x1e8403a0},
                                        std::endian::big,
                                        mem::access_size::half}) == 0 &&
                 !probe.read(
                     probe.context,
                     {{0x1e8403a2}, std::endian::big, mem::access_size::half}),
             "NPE-400 board probe reads zero; unknown registers fault"))
    return 1;
  if (!controller_configuration())
    return 1;
  if (!empty_port_adapters())
    return 1;
  if (!serial_board_identity())
    return 1;
  if (!io_console())
    return 1;
  if (!mapped_memory())
    return 1;
  try {
    if (argc != 2)
      return 1;
    if (!rom_contract() || !composition() || !archive_contract() ||
        !byte_memory_contract() || !rom_service_queues() || !real_rom(argv[1]))
      return 1;
    std::println("machine: ELF loading, chassis composition, ROM protection, "
                 "reset and real C7200 boot boundary passed");
    return 0;
  } catch (const std::exception &error) {
    std::println(stderr, "machine test: {}", error.what());
    return 1;
  }
}
