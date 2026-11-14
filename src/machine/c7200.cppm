module;

#include <array>
#include <cstdint>

export module machine.c7200;

import machine.core;
import devices.gt64120;

namespace dynamips::c7200 {
// Register windows from common/dev_c7200.c; unattached windows fault on access.
inline constexpr std::array windows{
    machine::device_window{"gt64120", {{0x14000000}, 0x1000}},
    machine::device_window{"rom-services", {{0x16000000}, 0x1000}},
    machine::device_window{"bootflash", {{0x1a000000}, 8 * 1024 * 1024}},
    machine::device_window{"nvram", {{0x1e000000}, 128 * 1024}},
    machine::device_window{"midplane-fpga", {{0x1e800000}, 0x1000}},
    machine::device_window{"io-fpga", {{0x1e840000}, 0x1000}},
    machine::device_window{"pci-io", {{0x100000000ULL}, 0x1000}}};

export inline constexpr machine::window_id gt64120{0};
export inline constexpr machine::window_id rom_services{1};
export inline constexpr machine::window_id bootflash{2};
export inline constexpr machine::window_id nvram{3};
export inline constexpr machine::window_id midplane_fpga{4};
export inline constexpr machine::window_id io_fpga{5};
export inline constexpr machine::window_id pci_io{6};

// Board images from common/dev_c7200.c.
export inline constexpr std::array<std::uint16_t, 64> npe400_eeprom{
    0x04FF, 0x4001, 0xF841, 0x0100, 0xC046, 0x0320, 0x001F, 0xC802,
    0x8249, 0x14BC, 0x0242, 0x4230, 0xC18B, 0x3131, 0x3131, 0x3131,
    0x3131, 0x0000, 0x0004, 0x0002, 0x0285, 0x1C0F, 0xF602, 0xCB87,
    0x4E50, 0x452D, 0x3430, 0x3080, 0x0000, 0x0000, 0xFFFF, 0xFFFF,
    0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
    0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
    0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
    0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
};
export inline constexpr std::array<std::uint16_t, 32> vxr_eeprom{
    0x0106, 0x0201, 0xff10, 0x45C5, 0x4906, 0x0303, 0xFFFF, 0xFFFF,
    0xFFFF, 0x0400, 0x0000, 0x0000, 0x4C09, 0x10B0, 0xFFFF, 0x00FF,
    0x0000, 0x0000, 0x6335, 0x8B28, 0x631D, 0x0000, 0x608E, 0x6D1C,
    0x62BB, 0x0000, 0x6335, 0x8B28, 0x0000, 0x0000, 0x6335, 0x8B28,
};

export inline constexpr std::array npe400_bridges{
    devices::pci_bridge_definition{0, 1, devices::bridge_model::dec21050},
    devices::pci_bridge_definition{0, 7, devices::bridge_model::dec21150},
    devices::pci_bridge_definition{0, 8, devices::bridge_model::dec21150},
    devices::pci_bridge_definition{0, 9, devices::bridge_model::dec21150},
    devices::pci_bridge_definition{1, 7, devices::bridge_model::dec21150},
    devices::pci_bridge_definition{1, 8, devices::bridge_model::dec21150},
    devices::pci_bridge_definition{1, 9, devices::bridge_model::dec21150}};

export inline constexpr std::uint16_t configuration_register = 0x2102;

export inline constexpr machine::definition npe400{
    "c7200-npe400",
    {"R7000",
     0x00002721,
     {0xffffffffbfc00000ULL},
     {0xffffffff80004000ULL},
     0x20000000},
    256 * 1024 * 1024,
    {{0x1fc00000}, 4 * 1024 * 1024},
    windows};
} // namespace dynamips::c7200
