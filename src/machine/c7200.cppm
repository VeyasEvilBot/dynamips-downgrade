module;

#include <array>
#include <cstdint>

export module machine.c7200;

import machine.core;

namespace dynamips::c7200 {
// Register windows from common/dev_c7200.c; unattached windows fault on access.
inline constexpr std::array windows{
    machine::device_window{"gt64120", {{0x14000000}, 0x1000}},
    machine::device_window{"rom-services", {{0x16000000}, 0x1000}},
    machine::device_window{"bootflash", {{0x1a000000}, 8 * 1024 * 1024}},
    machine::device_window{"nvram", {{0x1e000000}, 128 * 1024}},
    machine::device_window{"midplane-fpga", {{0x1e800000}, 0x1000}},
    machine::device_window{"io-fpga", {{0x1e840000}, 0x1000}}};

export inline constexpr machine::window_id gt64120{0};
export inline constexpr machine::window_id rom_services{1};
export inline constexpr machine::window_id nvram{3};
export inline constexpr machine::window_id io_fpga{5};

export inline constexpr std::uint16_t configuration_register = 0x2102;

export inline constexpr machine::definition npe400{
    "c7200-npe400",
    {"R7000", 0x00002721, {0xffffffffbfc00000ULL}, {0xffffffff80004000ULL}},
    256 * 1024 * 1024,
    {{0x1fc00000}, 4 * 1024 * 1024},
    windows};
} // namespace dynamips::c7200
