# Dynamips embedding experiment

[![Go Reference](https://pkg.go.dev/badge/github.com/tethux/dynamips-downgrade.svg)](https://pkg.go.dev/github.com/tethux/dynamips-downgrade)

This fork gives Tethux a typed C, C++23, and Go API for the stable Dynamips
emulator. The public Go package is at the repository root. The old TCP
hypervisor remains as a separate target for behavior comparisons; the
embedding library does not link against its command handlers. The required,
deferred, and legacy-only API scope lives in
[`bindings/schema/`](bindings/schema/).

This is an experimental, AI-assisted learning project. AI agents helped with
the binding backfill, refactor, and tests. I am using it to learn the emulator
and improve my development workflow. I do not claim this work is better than
the years of engineering behind Dynamips. I am having fun, and my long-term
goal is to rewrite most of the code into a smaller emulator containing only
what I need.

Christophe Fillot created Dynamips. Fabien Devaux, MtvE, Gordon Russell,
Jeremy Grossmann, Flávio J. Saraiva, GNS3, and other contributors maintained
and extended it. Daniel Lintott converted an earlier README to Markdown. The
repository includes [`LICENSE`](LICENSE); individual source files also carry
their own notices.

## Build and test

The project uses xmake, C23 and C++23. Release builds use Clang 19 and
libstdc++ 14 on Ubuntu 24.04. Install libelf and libpcap development files to
build the native code. Then run:

```sh
mise run build
mise run test
mise run lint:go
mise run lint:cpp
```

`mise run test` builds the native libraries and legacy executable, runs every
registered xmake test (C++ module lifecycle, JIT and guest memory), then runs the
Go tests against the freshly installed libraries, including legacy parity.
Native tests print a pass/fail summary and show diagnostics on failure. Register
new native suites with `add_tests` in their xmake target to include them here.
Use `test:native`, `test:jit`, `test:memory` or `test:go` for a focused run;
`test:core` runs JIT and memory together. Benchmarks remain separate, and IOS boot
tests require `DYNAMIPS_IOS_ARCHIVE` (see `test:ios` below).
`mise run lint:cpp` runs cppcheck on the new C/C++ embedding boundary
and clang-tidy on `bindings/*.cpp`. It requires `clang-tidy`; mise provides
cppcheck. The C++23 module partitions are checked by the Clang build because
xmake's compilation database does not currently include their module commands
for clang-tidy. xmake compiles with 14 parallel jobs by default here; set
`DYNAMIPS_BUILD_JOBS` to change the task setting. Use `mise run fmt:go` for Go
formatting and tidying both Go modules. `mise run lint:cpp:ports` checks the
migrated utility, packet, UDP and console C++ sources with strict
conversion/pedantic warnings as errors
and focused Clang analysis. Run `mise run compiledb` to prepare clangd.

## JIT playground

`mise run run:jit` runs a native-compiled MIPS program that writes `hello` through
an MMIO console, reads its ready status, and stores and loads a RAM word.
Use `mise run run:jit -- --dump-ir --registers` to inspect IR and final registers.
Hexadecimal instruction words may follow the options to replace the example.

The harness loads and fetches big-endian code at `0x1000`, maps data RAM at
`0x2000`, and maps console word registers at `0x8000` (TX) and `0x8004` (ready).
The console captures up to 1024 bytes; writing when full raises a device fault.
Programs are straight-line blocks of at most 1024 instructions. Supported
instructions are ORI, XOR, DADDU, LW, SW, LD and SD. Faults report the guest PC
and address, retain earlier effects, and stop later instructions.

Memory IR currently calls typed C++ helpers; inline JIT RAM-cache probing and
architectural MIPS exceptions remain future work.

## Go consumers

Native binaries are distributed as Linux amd64 GitHub release assets, not in
the Go module. Install a C/C++ compiler, pkg-config, and libelf, libpcap and
libnsl development libraries. Download the matching native release:

```sh
version=0.2.28
curl -fLO "https://github.com/tethux/dynamips-downgrade/releases/download/v${version}/dynamips-downgrade_${version}_linux_amd64.tar.gz"
curl -fLO "https://github.com/tethux/dynamips-downgrade/releases/download/v${version}/checksums.txt"
sha256sum --ignore-missing -c checksums.txt
mkdir -p "$HOME/.local/dynamips/$version"
tar -xzf "dynamips-downgrade_${version}_linux_amd64.tar.gz" -C "$HOME/.local/dynamips/$version"
export PKG_CONFIG_PATH="$HOME/.local/dynamips/$version/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
go get github.com/tethux/dynamips-downgrade@v0.2.28
```

Keep `PKG_CONFIG_PATH` set when building with `CGO_ENABLED=1`. Go downloads
source only; it does not install native dependencies. Consumers do not need
Xmake or a sibling checkout. The native static archives link against the
system C++ runtime, libelf, libpcap and libnsl, so the final executable may
still use shared system libraries. Match the Go and native release versions;
the host needs glibc and a C++ runtime compatible with the Ubuntu 24.04 release
build. After switching native installations, use `go build -a` or
`go test -a -count=1`: Go may otherwise reuse cached cgo output.

The [standalone consumer example](examples/consumer/) fetches the published
module without a local replacement. Other native platforms require a source
build with Xmake.

`mise run package:native` stages release libraries, headers and pkg-config
metadata in `build/release`. `mise run release:check` validates GoReleaser;
`mise run release:snapshot` builds and tests the native package without
publishing. Pushing a new `v*` tag publishes the archive and checksums through
GoReleaser, following the same release workflow as Shitnet. Generated native
binaries and release output are ignored by Git.

## Native install

The file `pkgconfig/dynamips-bindings.pc` tells `pkg-config` where the installed
headers and archives are and which libraries a final link needs. Xmake installs
it alongside both static archives:

```sh
mise exec -- xmake f -m release -y
mise exec -- xmake build -y dynamips-bindings
mise exec -- xmake install -o /path/to/prefix dynamips-bindings dynamips-core
PKG_CONFIG_PATH=/path/to/prefix/lib/pkgconfig pkg-config --cflags --libs dynamips-bindings
```

Set `PKG_CONFIG_PATH` to that directory for native consumers using the
installed archives. Go consumers use the same pkg-config package. Validate real IOS/JIT execution
when changing compiler or wrapper settings: a Nix Clang 21 source build exposed
an existing amd64 JIT register-preservation issue despite passing compilation
and provider unit tests. The published v0.2.28 asset passed the full Tethux IOS
lifecycle, persistence and networking suite in Nix.

## Go API

The Go package initializes Dynamips without CLI options or a TCP hypervisor.
It exposes VM lifecycle and configuration, cards, NIO, and C7200 settings.
The embedded runtime is process wide. Delete native objects when finished;
`Close` only releases a handle reference. Release all handles before closing
the runtime.

`VMConfig.StateDirectory` selects an existing absolute directory for that VM's
NVRAM, flash, ROMMON variables, and other generated files. Lease a separate
directory to each active VM. VM deletion retains saved state and removes its
lock; the caller owns the directory lifetime. Empty uses the legacy filenames
in the process working directory. IOS image paths remain separate.

See the [C7200 example](examples/basic/main.go) for a complete VM lifecycle,
including shutdown and error cleanup.

API reference: [dynamips](https://pkg.go.dev/github.com/tethux/dynamips-downgrade) ·
[errs](https://pkg.go.dev/github.com/tethux/dynamips-downgrade/errs) ·
[example command](https://pkg.go.dev/github.com/tethux/dynamips-downgrade/examples/basic).
Read the current API locally with:

```sh
mise exec -- go doc github.com/tethux/dynamips-downgrade
mise exec -- go doc github.com/tethux/dynamips-downgrade/errs
```

The [C7200 example](examples/basic/README.md) takes a local IOS image or zip
archive and opens its console:

```sh
DYNAMIPS_IOS_IMAGE=/path/to/c7200-ios.zip mise run run:example
```

The same terminal opens the IOS console. Press Enter after IOS boots, type
commands there, then press Ctrl+] and enter `quit` when finished.

## IOS integration and current performance

[`ios_boot_test.go`](internal/binding/ios_boot_test.go) boots a C7200 image
through both the embedded Go API and the old TCP hypervisor. It verifies IOS
commands, interface output, a hostname change and the resulting running config.
Cisco images are not distributed here:

```sh
DYNAMIPS_IOS_ARCHIVE=/path/to/c7200-ios.zip DYNAMIPS_IOS_RAM_MB=256 \
  mise run test:ios
```

The task tests freshly built amd64 native libraries through pkg-config, with
Go's build cache bypassed. It runs three sequential paired trials, alternating
API order. Set `DYNAMIPS_IOS_TRIALS=1..5` to change the count and
`DYNAMIPS_IOS_DETAILS=1` for ranges and all measurements. Each embedded VM has
fresh state and is stopped and deleted between trials.

The compact report shows median startup, IOS-ready and CLI times, CPU usage,
process RSS and Go memory. Go measurements include heap objects, runtime backing,
allocations and GC pauses; the detailed report also includes stacks, reserved
memory, allocation counts and goroutines. Go includes the test/console client.
Runtime backing (`Sys - HeapReleased`) is an estimate, not measured RSS. Total
RSS includes native emulator allocations and mappings. It cannot be split into
native and Go resident memory by subtracting Go runtime statistics.

Three pairs using the local 12.4(24)T5 archive passed on 2026-10-21. Median Go
heap objects were 2.38 MiB after CLI checks, runtime backing was 10.78 MiB and
total embedded RSS was 236.68 MiB. These are local observations: repeated
embedded runs share allocator caches, RSS peaks are sampled every 100 ms, and
legacy startup includes process launch. Both APIs use the same native core.

Real UDP Ethernet/IOS ARP and ICMP traffic, console reconnects and active
shutdown were also verified during the Asio migrations. Focused native tests
passed ASan/UBSan and relevant TSan checks. Full emulator UBSan remains limited
by existing JIT operations; a boot alone does not prove networking or teardown.

## Roadmap

The bindings are a bridge to a smaller emulator. `unstable/` is already gone,
and the old TCP hypervisor is isolated from the embedding library. Possible
next steps are:

1. Retire the hypervisor after the required API and IOS parity checks cover
   the workflows Tethux uses. Drop unused host JIT targets as the supported
   host and guest platforms become explicit.
2. Extend Asio only where remaining backends need it. UDP NIO reception and
   native console transport already use asynchronous Asio I/O; console device
   notifications remain on the native console thread. Timers and periodic tasks
   already have C++ implementations. Keep CPU execution separate from I/O.
3. Continue bounded utility migrations where a real consumer benefits. Registry,
   ROMMON, host file/alignment helpers and pure packet/address/checksum helpers
   already expose C++ mechanisms with compatible C boundaries. Retained C code
   and constant tables are not migration targets just to reduce file counts.
4. Make Cisco chassis, slots, cards, IRQ routing, and common FPGA register
   layouts mostly data-driven. Share PCI, MMIO, memory-map, and device
   lifecycle infrastructure, but keep genuinely different chip behavior in
   separate implementations.
5. Rewrite the remaining CPU, device, and networking code in tested slices.
   Consider a common guest IR and a maintained code emitter only after the
   supported JIT paths are measured. End with directories organized by
   runtime, CPU, devices, network, Cisco platforms, and console rather than
   the historical `common/` and `stable/` split.

## Architecture checks

`mise run lint:architecture` checks `.repolint-architecture.toml` using the sibling
`../repolint` checkout. Set `REPOLINT_SOURCE` for another checkout or `REPOLINT_BIN`
for a built checker containing the new rules. `mise run check` combines these
contracts with the existing Go policy, lint, and binding test tasks. Required
rules cannot be disabled through the configuration.

The published Go tool still reads `.repolint.toml`. The architecture checker
needs a repolint release before its rules can join the published-tool CI gate.
