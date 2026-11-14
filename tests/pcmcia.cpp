#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <print>
#include <string_view>
#include <dynamips/macros.h>
import devices.clpd6729;
import devices.gt64120;
import jit.memory;
import jit.devices;

namespace {
namespace mem = jit::memory;
namespace dev = dynamips::devices;
fn require(bool good, std::string_view message) -> void {
  if (!good) {
    std::println(stderr, "PCMCIA test: {}", message);
    std::exit(1);
  }
}

struct fixture {
  dev::clpd6729 socket;
  dev::gt64120 controller{{0x14000000}};
  mem::device &pci = *controller.descriptor().endpoint;
  mem::device &ports = *socket.descriptor().endpoint;
  static constexpr std::uint32_t address = 0x80028000;

  fixture() {
    const std::array functions{socket.configuration()};
    require(controller.configure_functions(functions).has_value(),
            "register bounded PCI function");
  }

  fn select_config(std::uint32_t value, bool root = false) -> void {
    require(pci.write(pci.context,
                      {{root ? 0x14000cf0ULL : 0x14000cf8ULL},
                       std::endian::little,
                       mem::access_size::word},
                      value)
                .has_value(),
            "select PCI configuration address");
  }
  fn read_config(unsigned reg) -> std::uint32_t {
    select_config(address | reg);
    const let value =
        pci.read(pci.context,
                 {{0x14000cfc}, std::endian::little, mem::access_size::word});
    require(value.has_value(), "read PCI configuration data");
    return static_cast<std::uint32_t>(*value);
  }
  fn write_config(unsigned reg, std::uint32_t value) -> void {
    select_config(address | reg);
    require(
        pci.write(pci.context,
                  {{0x14000cfc}, std::endian::little, mem::access_size::word},
                  value)
            .has_value(),
        "write PCI configuration data");
  }
  fn index(unsigned value) -> void {
    require(
        ports
            .write(
                ports.context,
                {{0x100000403ULL}, std::endian::little, mem::access_size::byte},
                value)
            .has_value(),
        "select socket register through MIPS PCI lane");
  }
  fn read(unsigned reg) -> std::uint64_t {
    index(reg);
    const let value = ports.read(
        ports.context,
        {{0x100000402ULL}, std::endian::little, mem::access_size::byte});
    require(value.has_value(), "read socket register");
    return *value;
  }
  fn write(unsigned reg, unsigned value) -> void {
    index(reg);
    require(
        ports
            .write(
                ports.context,
                {{0x100000402ULL}, std::endian::little, mem::access_size::byte},
                value)
            .has_value(),
        "write socket register");
  }
};

fn configuration() -> void {
  fixture f;
  require(f.read_config(0) == 0x11001013 && f.read_config(8) == 0x06050000 &&
              f.read_config(4) == 0x04000080 && f.read_config(0x10) == 1,
          "CL-PD6729 identity, class, reset command and unassigned BAR");
  require(!f.ports.read(
              f.ports.context,
              {{0x100000403ULL}, std::endian::little, mem::access_size::byte}),
          "PCI I/O enable controls operation-register decoding");
  f.write_config(4, 0xffffffff);
  require(f.read_config(4) == 0x040001c3,
          "only implemented command bits are writable");
  f.write_config(0x3c, 0xffffffff);
  require(f.read_config(0x3c) == 0x1ff,
          "interrupt line writable while pin stays INTA");
  f.write_config(0, 0);
  require(f.read_config(0) == 0x11001013, "PCI identity is read-only");
  f.write_config(0x10, 0xffffffff);
  require(f.read_config(0x10) == 0xfffffffd,
          "BAR probe describes four-byte I/O register aperture");
  require(!f.ports.read(
              f.ports.context,
              {{0x100000403ULL}, std::endian::little, mem::access_size::byte}),
          "BAR probing cannot retain stale port decoding");
  f.write_config(0x10, 0x501);
  require(f.read_config(0x10) == 0x501 &&
              !f.ports.read(f.ports.context, {{0x100000403ULL},
                                              std::endian::little,
                                              mem::access_size::byte}) &&
              f.ports.read(f.ports.context, {{0x100000503ULL},
                                             std::endian::little,
                                             mem::access_size::byte}) == 0,
          "BAR relocation changes decoding within the physical I/O page");
  const std::array duplicate{f.socket.configuration(),
                             f.socket.configuration()};
  require(!f.controller.configure_functions(duplicate) &&
              f.read_config(0) == 0x11001013,
          "failed PCI update retains existing function mapping");
  for (const std::uint32_t absent : {0x80038000u, 0x80028100u, 0x80028800u}) {
    f.select_config(absent);
    require(f.pci.read(f.pci.context, {{0x14000cfc},
                                       std::endian::little,
                                       mem::access_size::word}) == 0xffffffff,
            "different bus/device/function does not alias the PCMCIA function");
  }
  f.select_config(f.address, true);
  require(f.pci.read(f.pci.context, {{0x14000cf4},
                                     std::endian::little,
                                     mem::access_size::word}) == 0xffffffff,
          "second PCI root does not alias the PCMCIA function");
}
fn empty_sockets() -> void {
  fixture f;
  f.write_config(0x10, 0x401);
  f.write_config(4, 1);
  require(f.read(0) == 0x82 && f.read(0x40) == 0x82,
          "compatibility identity advertises memory and I/O interfaces");
  require(
      f.read(1) == 0x80 && f.read(0x41) == 0x80 && f.read(3) == 0,
      "empty sockets have no inserted card, power or unsolicited interrupt");
  f.write(2, 0x10);
  require(
      f.read(1) == 0xc0 && f.read(0x41) == 0x80,
      "manual socket power is independent from card detect and other socket");
  f.write(2, 0x30);
  require(f.read(1) == 0x80, "auto-power stays off without an inserted card");
  f.write(3, 0x45);
  f.write(0x43, 0x40);
  require(f.read(3) == 0x45 && f.read(0x43) == 0x40,
          "socket control registers retain independent state");
  f.write(4, 0xff);
  require(f.read(4) == 0 && f.read(4) == 0,
          "software cannot manufacture card-status-change events");
  f.write(0x17, 0x55);
  require(f.read(0x17) == 0xd5,
          "empty FIFO status remains set alongside scratchpad bits");
  f.write(0x1f, 0xff);
  require(f.read(0x1f) == 0xc0 && f.read(0x5f) == 0 && f.read(0x1f) == 0xc0,
          "chip-identification read side effects are shared across sockets");
  f.write(0x2e, 0);
  f.write(0x2f, 0xaa);
  f.write(0x6e, 0);
  f.write(0x6f, 0xbb);
  require(f.read(0x2f) == 0xaa && f.read(0x6f) == 0xbb,
          "extended scratchpad storage is selected per socket");
  f.index(0x27);
  require(!f.ports.read(
              f.ports.context,
              {{0x100000402ULL}, std::endian::little, mem::access_size::byte}),
          "reserved operation registers remain unsupported");
  f.index(0x80);
  require(!f.ports.read(
              f.ports.context,
              {{0x100000402ULL}, std::endian::little, mem::access_size::byte}),
          "reserved socket index cannot escape bounded storage");
  require(!f.ports.read(f.ports.context, {{0x100000400ULL},
                                          std::endian::little,
                                          mem::access_size::byte}) &&
              !f.ports.read(f.ports.context, {{0x100000402ULL},
                                              std::endian::little,
                                              mem::access_size::word}),
          "unmodeled ports and wide transfers return typed failures");
  f.socket.reset();
  require(f.read_config(4) == 0x04000080 && f.read_config(0x10) == 1,
          "reset restores PCI state and disables I/O decoding");
  f.write_config(0x10, 0x401);
  f.write_config(4, 1);
  require(f.read(2) == 0 && f.read(3) == 0 && f.read(0x17) == 0x80 &&
              f.read(0x3a) == 1 && f.read(0x3b) == 5 && f.read(0x3e) == 0x14 &&
              f.read(0x1e) == 2,
          "reset restores socket and timing defaults");
}
fn collisions() -> void {
  dev::clpd6729 socket;
  dev::gt64120 controller{{0x14000000}};
  const std::array bridges{
      dev::pci_bridge_definition{0, 1, dev::bridge_model::dec21050}};
  let function = socket.configuration();
  function.bus = 0;
  function.device = 1;
  const std::array functions{function};
  require(controller.configure_bridges(bridges).has_value(), "install bridge");
  require(!controller.configure_functions(functions),
          "function cannot shadow a bridge");
  require(controller.configure_bridges({}).has_value() &&
              controller.configure_functions(functions).has_value(),
          "replace stopped configuration");
  require(!controller.configure_bridges(bridges),
          "bridge cannot shadow a function");
}
} // namespace

fn main() -> int {
  configuration();
  collisions();
  empty_sockets();
  std::println("PCMCIA: PCI routing, BARs, empty sockets, register state and "
               "reset passed");
}
