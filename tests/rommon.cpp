#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <optional>
#include <print>
#include <string>
#include <string_view>
#include <dynamips/macros.h>
import devices.rom_services;
import jit.devices;
import jit.memory;

namespace {
namespace mem = jit::memory;
fn require(bool good, std::string_view message) -> void {
  if (!good) {
    std::println(stderr, "ROMMON test: {}", message);
    std::exit(1);
  }
}

struct fixture {
  dynamips::devices::rom_services device{{0x16000000}};
  mem::device &endpoint = *device.descriptor().endpoint;
  fn write(unsigned offset, unsigned value) -> void {
    require(endpoint
                .write(endpoint.context,
                       {{0x16000000 + offset},
                        std::endian::big,
                        mem::access_size::word},
                       value)
                .has_value(),
            "write ROMMON service register");
  }
  fn read(unsigned offset) -> std::uint32_t {
    const let value = endpoint.read(
        endpoint.context,
        {{0x16000000 + offset}, std::endian::big, mem::access_size::word});
    require(value.has_value(), "read ROMMON service register");
    return static_cast<std::uint32_t>(*value);
  }
  fn send(std::string_view text) -> void {
    write(0x58, 3);
    for (const char byte : text)
      write(0x54, static_cast<unsigned char>(byte));
    write(0x54, 0);
  }
  fn set(std::string_view assignment) -> bool {
    send(assignment);
    write(0x58, 1);
    return read(0x58) == 0;
  }
  fn get(std::string_view name) -> std::optional<std::string> {
    send(name);
    write(0x58, 2);
    if (read(0x58))
      return {};
    std::string value;
    for (unsigned i = 0; i < 512; i++) {
      const let byte = read(0x54);
      if (!byte)
        return value;
      value.push_back(static_cast<char>(byte));
    }
    require(false, "ROMMON values remain NUL terminated");
    return {};
  }
};

fn variables() -> void {
  fixture f;
  require(!f.get("missing") && f.set("BOOT=bootflash:image.bin") &&
              f.get("BOOT") == "bootflash:image.bin",
          "new variables are readable through the ROM data protocol");
  require(f.set("BOOT=new=value") && f.get("BOOT") == "new=value",
          "update splits only the first equals sign");
  require(f.set("EMPTY=") && f.get("EMPTY") == "" && !f.get("absent"),
          "empty values remain distinct from absent names");
  require(!f.set("=value") && !f.set("bad-format") && !f.set(""),
          "malformed assignments report guest-visible failure");
  require(!f.set(std::string("BOOT=") + std::string(600, 'x')) &&
              f.get("BOOT") == "new=value",
          "overflow rejects the whole update and preserves the old value");
  for (const std::size_t size : {510uz, 511uz}) {
    const std::string value(size - 5, 'x');
    require(f.set("LONG=" + value) && f.get("LONG") == value,
            "full-length assignments accept an explicit NUL terminator");
  }
  require(!f.set("LONG=" + std::string(507, 'y')) &&
              f.get("LONG") == std::string(506, 'x'),
          "one extra data byte rejects the update atomically");
  f.device.reset();
  require(f.read(0x58) == 0 && f.read(0x54) == 0 &&
              f.get("BOOT") == "new=value" && f.get("EMPTY") == "",
          "warm reset clears the transaction while retaining variable values");
}
fn bounded_store() -> void {
  fixture f;
  for (std::size_t i = 0; i < f.device.variable_capacity; i++)
    require(f.set("V" + std::to_string(i) + "=value"),
            "fill bounded variable slots");
  require(!f.set("EXTRA=value") && f.get("V0") == "value" &&
              f.set("V0=updated") && f.get("V0") == "updated",
          "full table rejects new names while allowing existing updates");
}
} // namespace

fn main() -> int {
  variables();
  bounded_store();
  std::println("ROMMON: variable updates, query, empty values, bounds and warm "
               "reset passed");
}
