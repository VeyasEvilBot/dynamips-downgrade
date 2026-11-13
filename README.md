# Dynamips emulator rewrite

[![Go Reference](https://pkg.go.dev/badge/github.com/tethux/dynamips-downgrade.svg)](https://pkg.go.dev/github.com/tethux/dynamips-downgrade)

This is an experimental rewrite of Dynamips into a small, composable C++23
emulator for Tethux. The aim is a slim C++ core with native modules, a shared
MIPS IR, and an AsmJit backend. A roughly 20k-line emulator is a design ambition,
not the current size or a reason to sacrifice correct device behavior.

Tethux manages workloads and virtualization orchestration. Dynamips executes
the guest and models its hardware:

| Responsibility | Owner |
| --- | --- |
| Topology, inventory, placement and provider orchestration | Tethux |
| Storage-directory leases, port allocation and external Ethernet fabric | Tethux |
| CPU execution, guest memory, address translation and interrupts | Dynamips |
| Chassis, cards, devices, console and device-attached network endpoints | Dynamips |
| Native machine lifetime and reading/writing native state files | Dynamips |

Tethux chooses storage locations and connects workloads. Native machine objects
still need explicit creation, reset, stop and teardown inside the emulator.
The rewrite does not aim to reproduce the old TCP hypervisor's orchestration.

Two implementations coexist. The new core in `src/` is a standalone bring-up
path. The existing C/C++ core and root Go package remain the current Tethux
integration; the Go bindings have **not** switched to the new JIT. Their public
ownership contracts are documented in [`bindings/README.md`](bindings/README.md)
and [`bindings/schema/`](bindings/schema/).

This is an AI-assisted learning project. AI agents helped with the binding
backfill, refactor and tests. I am using it to learn the emulator and improve my
development workflow. I do not claim this work is better than the years of
engineering behind Dynamips.

Christophe Fillot created Dynamips. Fabien Devaux, MtvE, Gordon Russell,
Jeremy Grossmann, Flávio J. Saraiva, GNS3, and other contributors maintained
and extended it. Daniel Lintott converted an earlier README to Markdown. The
repository includes [`LICENSE`](LICENSE); individual source files also carry
their own notices.

## Rewrite checkpoint

Verified on Linux amd64 on 2026-11-13 with a C7200 NPE-400 and IOS
12.4(24)T5: the guest decompresses its image, hands off to the IOS kernel,
sets up mappings and prints:

```text
Cisco IOS Software, 7200 Software (C7200-ADVENTERPRISEK9-M), Version 12.4(24)T5, RELEASE SOFTWARE (fc3)
```

This is a boot checkpoint, not a complete IOS boot, working CLI or network
validation. The next observed stop is an unimplemented I/O FPGA DS1620
temperature-control access at physical address `0x1e84020c`, guest PC
`0x60216a70`. Missing PCMCIA identification also produces a warning before the
banner. Cisco images are not distributed here.

| Area | Implemented in the new core | Remaining limits |
| --- | --- | --- |
| CPU/JIT | MIPS64 integer operations, branches and delay slots, HI/LO, CP0, LL/SC, COP1 bit and memory transfers | Incomplete ISA; no full floating-point implementation |
| IR | Compact 16-byte nodes, bounded storage, readable printing and optimization passes | More lowering and optimization work remains |
| Memory | Big-endian RAM/MMIO, two-level translation cache, architectural TLB setup/probe, batched invalidation | Complete guest TLB exception/refill behavior remains |
| Code dispatch | Native AMD64 blocks, writable guest code, byte-change checks and CACHE maintenance | More profiling and architectural validation needed |
| Machine | ELF/ZIP loading, bundled ROM, NPE-400 geometry, ROMMON variables | Boot bring-up only; persistence parity is not established |
| Devices | Queued SCN2681 console/aux, EEPROM reads, FPGA startup registers, GT64120 PCI configuration and built-in bridges | DS1620, PCMCIA, DMA, cards and packet delivery remain |
| Time | CP0 Count/Compare and eligible interrupt delivery at block boundaries | Virtual instruction-based timing, not cycle accuracy |

Translation-cache invalidation revalidates executable backing independently of
compiled-code reuse; byte comparisons still detect writes through aliases.
Native outcomes distinguish delivered overflow/syscall exceptions from host
failures and charge only their executed instruction prefix. Annulled
branch-likely delay slots do not advance virtual time. Memory dispatch
can compile shorter straight-line prefixes for small budgets; a branch and
its delay slot still require a budget of at least two. Memory faults still
stop bring-up rather than delivering complete architectural TLB exceptions.
CPU-owned segment translation, a physical-region bus, deterministic device
events/IRQ routing remain migration work. A bounded independent integer interpreter now executes cold blocks and provides differential checks.

The execution cache retains 8,192 blocks in four-way sets. Cold integer blocks
use an independent interpreter before native promotion; hot loops use bounded
native traces. RAM write versions, opaque-callback epochs and byte checks preserve
code coherency, including physical aliases. Checked native RAM probes retain
fault/MMIO fallbacks. Traces stop for instruction budgets, timer deadlines,
code changes and device-service boundaries.

On one Linux amd64 host, seven alternating process starts per core using the
same IOS 12.4(24)T5 image and CPU 2 measured median launch-to-banner times of
1.026 s for the new core and 1.111 s for legacy Dynamips (7.7% less time).
The previous rewrite checkpoint measured 2.89 s. These are host-specific boot
milestone measurements; CLI readiness and networking remain unvalidated.
See `benchmarks/README.md` for reproduction commands.

Unsupported operations report typed failures. Reaching the banner does not
prove that other images, networking, persistence or teardown are ready.

## Build and run

Use xmake, Clang with C++23 module support, and a C++23 standard library.
Install libelf and libpcap development files. The xmake configuration resolves
the other declared dependencies, including AsmJit, Asio, zlib and minizip.
The new native backend currently targets Linux amd64.

```nu
mise run run:c7200 -- --ios /path/to/c7200-ios.zip
```

This task builds an isolated release profile in `tools/boot-release`, leaving
the development configuration available for clangd. ELF images are also
accepted. `--stop-at-handoff` stops before kernel execution; `--budget N` limits
execution for diagnosis. Without a budget it runs until a stop or failure.
Use `run:c7200:dev` to run the current development configuration instead.

For development, prepare module consumers and the compilation database, then
run the focused checks for the behavior being changed:

```nu
mise run compiledb
mise run test:core
mise run test:console
mise run test:machine
mise run test:boot
mise run lint:cpp:jit ::: lint:clangd
```

`test:boot` rebuilds the CPU and machine consumers together before testing.
`test:core` covers JIT, memory and device suites; `test:console` uses real Unix
streams. Run the real IOS command after boot-path changes and record the next
concrete blocker. A boot does not replace packet, persistence or teardown tests.

`mise run test` builds the legacy libraries and executable, runs registered
native tests and runs Go tests against freshly installed native archives.
`mise run check` combines repository policy, lint and test tasks. Use focused
tasks during iteration; benchmarks remain separate. `mise run compiledb`
prepares the module compilation database used by clangd and Clang analysis.

## Core layout and execution contracts

`src/jit/` owns decoding, IR, optimization, native emission, dispatch and guest
memory. `src/machine/` composes the chassis, image loading and boot state.
`src/devices/` models hardware; `src/host/` supplies owned host resources and
Asio console transport. `modules/asmjit.cppm` wraps the emitter dependency.

CPU and device execution have one owner. Guest loads and stores use cached
translation and typed cold-path failures. Mapping owners can batch invalidations
before resuming execution. The architectural TLB is distinct from the host
translation caches. Memory IR currently calls C++ helpers; inline native RAM
cache probes remain future work.

Devices use bounded FIFOs and run service hooks between bounded execution
slices. MMIO callbacks do not perform host I/O. Asio transport stages bytes and
retains unwritten data through partial writes and backpressure. Shutdown must
settle callbacks before releasing their buffers. Device and memory mappings
borrow their backing objects, which must outlive execution.

The immutable playground dispatcher requires sealed executable backing and
rejects writable aliases. The chassis uses writable-code dispatch with bounded
byte comparisons and mapping invalidation. Direct host writes must obey the
selected dispatch and LL/SC reservation contracts; borrowed memory is not a
concurrently writable interface.

## JIT and device playground

The small harness exercises native MIPS translation, RAM and queued MMIO UART
without a Cisco image. Inspect its IR and registers, supply instruction words,
or run an RX-to-TX echo guest:

```nu
mise run run:jit -- --dump-ir --registers
mise run run:jit -- --echo "hello devices"
```

The default example is in `src/jit/example.cppm`. The harness maps code at
`0x1000`, RAM at `0x2000`, and console TX/status/RX word registers at
`0x8000`, `0x8004` and `0x8008`. TX and RX each have a bounded 1024-byte FIFO.
The shared FIFO has one owner, without concurrent producer/consumer support.

For a live Unix console, run the first command in one terminal and the second
in another:

```nu
mise run run:jit -- --console /tmp/dynamips-console.sock
socat STDIO UNIX-CONNECT:/tmp/dynamips-console.sock
```

The echo session finishes after the client half-closes input and queued output
is drained. Ctrl-C or SIGTERM cancels pending I/O. The listener refuses existing
paths and removes its own socket on shutdown. This playground session does not
support reconnecting clients.

## Legacy-core Go consumers

Native binaries are distributed as Linux amd64 GitHub release assets, not in
the Go module. Install a C/C++ compiler, pkg-config, and libelf, libpcap and
libnsl development libraries. Download the matching native release:

```nu
let version = "0.2.28"
curl -fLO $"https://github.com/tethux/dynamips-downgrade/releases/download/v($version)/dynamips-downgrade_($version)_linux_amd64.tar.gz"
curl -fLO $"https://github.com/tethux/dynamips-downgrade/releases/download/v($version)/checksums.txt"
sha256sum --ignore-missing -c checksums.txt
let prefix = ($nu.home-path | path join ".local" "dynamips" $version)
mkdir $prefix
tar -xzf $"dynamips-downgrade_($version)_linux_amd64.tar.gz" -C $prefix
$env.PKG_CONFIG_PATH = ($prefix | path join "lib" "pkgconfig")
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

```nu
mise exec -- xmake f -m release -y
mise exec -- xmake build -y dynamips-bindings
mise exec -- xmake install -o /path/to/prefix dynamips-bindings dynamips-core
with-env {PKG_CONFIG_PATH: /path/to/prefix/lib/pkgconfig} { pkg-config --cflags --libs dynamips-bindings }
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

```nu
mise exec -- go doc github.com/tethux/dynamips-downgrade
mise exec -- go doc github.com/tethux/dynamips-downgrade/errs
```

The [C7200 example](examples/basic/README.md) takes a local IOS image or zip
archive and opens its console:

```nu
with-env {DYNAMIPS_IOS_IMAGE: /path/to/c7200-ios.zip} { mise run run:example }
```

The same terminal opens the IOS console. Press Enter after IOS boots, type
commands there, then press Ctrl+] and enter `quit` when finished.

## Legacy-core IOS integration and performance

[`ios_boot_test.go`](internal/binding/ios_boot_test.go) boots a C7200 image
through both the embedded Go API and the old TCP hypervisor. It verifies IOS
commands, interface output, a hostname change and the resulting running config.
Cisco images are not distributed here:

```nu
with-env {DYNAMIPS_IOS_ARCHIVE: /path/to/c7200-ios.zip, DYNAMIPS_IOS_RAM_MB: "256"} { mise run test:ios }
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

## Next rewrite steps

1. Implement the DS1620 temperature-control behavior and PCMCIA identification,
   using the next real IOS stop to choose a bounded piece of work.
2. Extend missing CPU instructions, guest exceptions and device interrupts with
   focused semantic tests and real-image checks.
3. Add card and network-device behavior on the existing MMIO/service contracts,
   then verify actual packet delivery, shutdown and persistence.
4. Integrate the new core into Tethux through a small explicit lifecycle API
   once those contracts work. Keep topology and provider orchestration in Tethux.
5. Retire legacy implementations only after their required consumers use the
   replacement. Reuse useful C++ helpers; do not port utilities just to reduce
   the number of C files.

## Architecture checks

`mise run lint:architecture` checks `.repolint-architecture.toml` using the sibling
`../repolint` checkout. Set `REPOLINT_SOURCE` for another checkout or `REPOLINT_BIN`
for a built checker containing the new rules. `mise run check` combines these
contracts with the existing Go policy, lint, and binding test tasks. Required
rules cannot be disabled through the configuration.

The published Go tool still reads `.repolint.toml`. The architecture checker
needs a repolint release before its rules can join the published-tool CI gate.
