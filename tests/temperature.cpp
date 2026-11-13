#include <bit>
#include <cstdlib>
#include <expected>
#include <print>
#include <string_view>
#include <dynamips/macros.h>
import devices.c7200_iofpga;
import jit.memory;
import jit.devices;

namespace {
namespace mem = jit::memory;
fn require(bool good, std::string_view message) -> void {
  if (!good) {
    std::println(stderr, "temperature test: {}", message);
    std::exit(1);
  }
}

struct serial_bus {
  dynamips::devices::c7200_iofpga io;
  mem::device &endpoint = *io.descriptor().endpoint;
  unsigned select = 0;

  fn write(unsigned offset, unsigned value) -> void {
    require(endpoint
                .write(endpoint.context,
                       {{0x1e840000 + offset},
                        std::endian::big,
                        mem::access_size::word},
                       value)
                .has_value(),
            "write FPGA sensor register");
  }
  fn read() -> unsigned {
    const let value =
        endpoint.read(endpoint.context,
                      {{0x1e84022c}, std::endian::big, mem::access_size::word});
    require(value.has_value(), "read FPGA sensor data");
    return static_cast<unsigned>(*value);
  }
  fn clock(bool high) -> void { write(0x20c, select | (high ? 16u : 0u)); }
  fn send(unsigned value, unsigned bits) -> void {
    for (unsigned i = 0; i < bits; i++) {
      clock(false);
      write(0x214, (value >> i) & 1);
      write(0x214, (value >> i) & 1);
      clock(true);
      clock(true);
    }
  }
  fn command(unsigned value, unsigned sensors = 1) -> void {
    write(0x20c, 16);
    select = sensors;
    clock(true);
    send(value, 8);
  }
  fn receive(unsigned bits, unsigned sensor = 0) -> unsigned {
    unsigned value = 0;
    for (unsigned i = 0; i < bits; i++) {
      clock(false);
      const let pins = read();
      require(read() == pins, "repeated MMIO reads do not shift sensor data");
      value |= ((pins >> sensor) & 1) << i;
      clock(true);
      require(read() == 15, "sensors release DQ while CLK is high");
    }
    return value;
  }
  fn reg(unsigned command, unsigned bits = 9, unsigned sensor = 0)
      -> unsigned {
    this->command(command, 1u << sensor);
    return receive(bits, sensor);
  }
};

fn protocol() -> void {
  serial_bus bus;
  require(bus.read() == 15, "deselected sensors release their data pins");
  require(bus.reg(0xac, 8) == 0x8a,
          "power-on DONE, CPU and reserved config bits");
  for (unsigned sensor = 0; sensor < 4; sensor++)
    require(
        bus.reg(0xaa, 16, sensor) == 44,
        "all four sensors report signed half-degree data with zero padding");
  bus.command(1, 2);
  bus.send(190, 16);
  require(bus.reg(0xa1, 16, 1) == 190 && bus.reg(0xa1) == 0,
          "threshold writes select independent sensors and ignore padding");
  bus.command(2);
  bus.send(492, 9);
  bus.command(0xee);
  require(bus.reg(0xa2) == 492 && (bus.reg(0xac, 8) & 0x60) == 0x40,
          "negative nine-bit thresholds compare as signed temperatures");
  bus.command(1);
  bus.send(120, 9);
  bus.command(2);
  bus.send(60, 9);
  bus.command(0xee);
  require((bus.reg(0xac, 8) & 0x60) == 0x60,
          "threshold alarm flags latch across conversions");
  bus.command(0x0c);
  bus.send(3, 8);
  require(bus.reg(0xac, 8) == 0x8b,
          "configuration writes clear sticky alarms while preserving DONE");
  bus.command(0xee);
  require(bus.reg(0xac, 8) == 0xab,
          "new conversion relatches the active low alarm");
  bus.command(0x0c);
  bus.send(255, 8);
  require(bus.reg(0xac, 8) == 0xab,
          "NVB and reserved configuration bits are not software writable");
  bus.command(0x22);
  const let counter = bus.reg(0xa0);
  const let slope = bus.reg(0xa9);
  require(counter == 12 && slope == 16 && 4 * (slope - counter) == slope,
          "high-resolution counter and slope describe the same 22 C ambient");
  bus.command(1);
  bus.send(255, 4);
  require(bus.reg(0xa1) == 120, "RST aborts incomplete threshold writes");
  bus.io.reset();
  require(bus.reg(0xa1) == 120 && bus.reg(0xa2) == 60 &&
              (bus.reg(0xac, 8) & 3) == 3 && bus.reg(0xa1, 9, 1) == 190,
          "board reset preserves sensor nonvolatile registers");
  bus.write(0x214, 1);
  bus.write(0x20c, 16);
  bus.select = 1;
  bus.clock(true);
  for (unsigned i = 0; i < 8; i++) {
    bus.clock(false);
    if (i)
      bus.write(0x214, 0);
    bus.clock(true);
  }
  bus.send(70, 9);
  require(bus.reg(0xa1) == 70,
          "DQ driven before RST remains latched across deselection");
  bus.command(0xaa, 15);
  bus.clock(false);
  require(bus.read() == 0,
          "all four selected sensors drive the first temperature bit");
  require(!bus.endpoint.read(
              bus.endpoint.context,
              {{0x1e840230}, std::endian::big, mem::access_size::word}),
          "unimplemented neighboring registers still fail");
}
} // namespace

fn main() -> int {
  protocol();
  std::println("temperature: four sensors, clock edges, thresholds, config, "
               "reset and padding passed");
}
