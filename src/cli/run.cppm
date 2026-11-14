module;

#include <CLI/CLI.hpp>
#include <charconv>
#include <cstdint>
#include <optional>
#include <string>
#include <system_error>

#include <dynamips/macros.h>

export module cli.run;

import boot.application;

export namespace dynamips::cli {
class run_command {
public:
  fn attach(CLI::App &app, int &exit_code) -> void {
    let *command = app.add_subcommand("run", "Run a C7200 IOS image");
    command->add_option("IMAGE", image_, "IOS ELF image or ZIP archive")
        ->required()
        ->check(CLI::ExistingFile);
    command->add_option("--rom", rom_, "C7200 ROM microcode ELF")
        ->capture_default_str()
        ->check(CLI::ExistingFile);
    let *socket = command->add_option(
        "--console", console_,
        "Listen on a Unix console socket instead of the terminal");
    let *terminal = command->add_flag("--stdio", stdio_,
                                       "Use the terminal directly (default)");
    socket->excludes(terminal);
    socket->check(CLI::Validator{[](const std::string &value) -> std::string {
                                   return value.empty()
                                              ? "socket path must not be empty"
                                              : "";
                                 },
                                 "PATH"});
    command
        ->add_option("--budget", budget_,
                     "Stop after N guest instruction slots")
        ->check(CLI::Validator{[](const std::string &text) -> std::string {
                                 std::uint64_t value{};
                                 const let [end, error] = std::from_chars(
                                     text.data(), text.data() + text.size(),
                                     value);
                                 if (error != std::errc{} ||
                                     end != text.data() + text.size() || !value)
                                   return "must be a positive 64-bit integer";
                                 return {};
                               },
                               "POSITIVE"});
    command->add_flag(
        "--dump-state", dump_state_,
        "Print CPU state on guest exceptions and execution stops");
    command->add_flag("--dump-ir", dump_ir_,
                      "Print translated IR for diagnosis");
    command->add_flag("--stop-at-handoff", stop_at_handoff_,
                      "Stop when the ROM transfers execution to IOS");
    command->callback([this, &exit_code] {
      boot::options options;
      options.rom = rom_.c_str();
      options.ios = image_.c_str();
      options.budget = budget_;
      options.dump_ir = dump_ir_;
      options.stop_at_handoff = stop_at_handoff_;
      if (!console_.empty())
        options.console_path = console_;
      options.dump_state = dump_state_;
      options.stdio_console = console_.empty();
      exit_code = boot::run(options);
    });
  }

private:
  std::string image_;
  std::string rom_ = "stable/mips64_microcode";
  std::string console_;
  std::optional<std::uint64_t> budget_;
  bool stdio_ = false;
  bool dump_ir_ = false;
  bool dump_state_ = false;
  bool stop_at_handoff_ = false;
};
} // namespace dynamips::cli
