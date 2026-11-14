#include <CLI/CLI.hpp>
#include <exception>
#include <print>

#include <dynamips/macros.h>

import cli.run;

fn main(int argc, char **argv) -> int {
  try {
    CLI::App app{"C7200 emulator", "dynamips-downgrade"};
    app.require_subcommand(1, 1);
    int exit_code = 0;
    dynamips::cli::run_command run;
    run.attach(app, exit_code);
    try {
      app.parse(argc, argv);
      return exit_code;
    } catch (const CLI::ParseError &error) {
      return app.exit(error);
    }
  } catch (const std::exception &error) {
    std::println(stderr, "dynamips-downgrade: {}", error.what());
    return 1;
  }
}
