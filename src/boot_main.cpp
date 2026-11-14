#include <charconv>
#include <cstdint>
#include <exception>
#include <optional>
#include <print>
#include <string_view>
#include <system_error>
#include <dynamips/macros.h>
import boot.application;

fn main(int argc, char **argv) -> int {
  try {
    const char *rom = "stable/mips64_microcode";
    const char *ios = nullptr;
    std::optional<std::string_view> console_path;
    std::optional<std::uint64_t> budget;
    bool stdio_console = false;
    bool dump_ir = false;
    bool dump_state = false;
    bool stop_at_handoff = false;
    for (int i = 1; i < argc; i++) {
      const std::string_view argument{argv[i]};
      if (argument == "--help") {
        std::println(
            "Usage: c7200-boot [--rom ELF] [--ios ELF_OR_ZIP] [--budget N] "
            "[--stdio | --console PATH] [--dump-state] [--dump-ir] "
            "[--stop-at-handoff]\n"
            "Boot the C7200 ROM and IOS image.\n"
            "Default ROM: stable/mips64_microcode. No instruction limit unless "
            "--budget is set.\n"
            "--console listens on a raw Unix stream and waits for one client "
            "before running. Without it, console output goes to stdout.\n"
            "Exit 2 means boot is "
            "blocked or the instruction budget is exhausted.");
        return 0;
      }
      if (argument == "--stdio") {
        stdio_console = true;
        continue;
      }
      if (argument == "--console") {
        if (console_path || i + 1 == argc ||
            std::string_view{argv[i + 1]}.empty()) {
          std::println(stderr, "boot: --console requires one socket path");
          return 1;
        }
        i++;
        console_path = argv[i];
        continue;
      }
      if (argument == "--stop-at-handoff") {
        stop_at_handoff = true;
        continue;
      }
      if (argument == "--dump-state") {
        dump_state = true;
        continue;
      }
      if (argument == "--dump-ir") {
        dump_ir = true;
        continue;
      }
      if ((argument != "--rom" && argument != "--ios" &&
           argument != "--budget") ||
          i + 1 == argc) {
        std::println(stderr, "boot: invalid option or missing value: {}",
                     argument);
        return 1;
      }
      i++;
      if (argument == "--rom")
        rom = argv[i];
      else if (argument == "--ios")
        ios = argv[i];
      else {
        const std::string_view text{argv[i]};
        std::uint64_t limit = 0;
        const let parsed =
            std::from_chars(text.data(), text.data() + text.size(), limit);
        if (text.empty() || parsed.ec != std::errc{} ||
            parsed.ptr != text.data() + text.size()) {
          std::println(stderr, "boot: invalid instruction budget: {}", text);
          return 1;
        }
        budget = limit;
      }
    }
    return dynamips::boot::run({rom, ios, budget, dump_ir, stop_at_handoff,
                                console_path, dump_state, stdio_console});
  } catch (const std::exception &error) {
    std::println(stderr, "boot: {}", error.what());
    return 1;
  }
}
