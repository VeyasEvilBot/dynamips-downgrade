# Emulator rewrite

The goal is a small, composable C++23 emulator for Tethux. New CPU, memory,
JIT and device work belongs in `src/`, using native modules and the existing
AsmJit backend. The standalone C7200 boot path is the current integration
checkpoint; the public Go bindings still use the legacy core. Do not confuse
passing legacy integration tests with validating the new implementation.

Tethux in `../sme` owns topology, inventory, provider orchestration, storage
leases, port allocation, and the external Ethernet fabric. Read the relevant
Tethux consumer before choosing a replacement. Do not recreate these features
inside Dynamips or pursue legacy hypervisor parity.

Dynamips owns CPU execution, guest memory, interrupts, chassis, cards, device
behavior, native machine state and lifecycle, native state-file formats,
console endpoints, and NIO endpoints attached to devices. Tethux chooses
storage directories; Dynamips reads and writes native files within them.
Orchestration does not remove native object lifetimes or make `stable/vm.c`
redundant before its consumers have a replacement.

## Implementation

Read surrounding modules and `include/dynamips/macros.h`. Follow existing
`fn`, `let`, formatting and module conventions, use postfix increment, and
prefer RAII, expressive types, `std::expected` and `std::print`/`std::println`.
Keep modules bounded and reusable. Reuse useful `src/host/` mechanisms instead
of routing new C++ consumers through the C ABI. Keep compatibility wrappers
thin at the boundary. Do not port utilities or rewrite constant tables simply
to reduce the C file count. Keep legacy sources unless deletion is requested.

Keep guest load/store and dispatch paths short. Reserve bounded storage outside
execution loops where practical; measure before adding unchecked shortcuts.
Return typed cold-path failures for unsupported behavior. Do not hide missing
hardware with fabricated successful accesses just to advance boot. Retain
intentional register defaults only where the modeled hardware contract calls
for them.

CPU execution and device servicing have one owner. MMIO callbacks enqueue or
consume device state; host I/O belongs at service boundaries through the
existing Asio transport. Preserve FIFO backpressure and pending data. Borrowed
RAM, mapped devices and callback contexts must outlive their users. Respect
code-cache invalidation and LL/SC reservation rules when changing mappings or
writing guest memory. Virtual CPU time is separate from host I/O timing.

Keep existing public contracts. `bindings/README.md` and `bindings/schema/`
document ownership and API scope; they are references, not a new binding-work
assignment. Do not expand APIs or regenerate bindings incidentally. Audit
callers, ownership, locking and failure behavior before replacing a mechanism.
Keep C-facing declarations usable from C, preserve registry references and
externally serialized native calls, and stop workers before freeing callback
state. Do not call Go from native workers or let C++ exceptions cross the C ABI.

## Verification and history

Check `mise.toml` first and use its focused tasks. `mise run compiledb` prepares
clangd and module consumers. Use `test:core`, `test:console`, `test:machine` or
`test:boot` for the affected path, and `lint:cpp:jit` and `lint:clangd` for new
core code. Separate multiple mise tasks with `:::`. Resolve compiler and clangd
warnings. Use focused ASan/UBSan checks for ownership or low-level changes.

For CPU and boot work, implement a bounded feature, test its semantics, and
rerun the real IOS input when available. Use the next concrete failure as
feedback. A build or IOS banner does not prove CLI readiness, packet delivery,
persistence or teardown safety. Do not replace real integration checks with
mocks when the real input is available. When switching legacy implementations,
check xmake source selection and packaged native archives so Go tests exercise
the replacement. Benchmark the actual optimized path with representative input;
do not promise speedups from architecture alone.

Keep the public rewrite checkpoint and limitations in README.md current. Keep
detailed migration tables and investigation logs in locally excluded planning
files. Comments should be brief and explain something necessary; do not copy
legacy file-header boilerplate into new modules. Preserve required license and
attribution notices.

Use jj. Describe changes, create new changes, or push only when asked; existing
session authorization persists. When separate changes per feature are requested,
test before describing each coherent step. Describing does not imply moving to
a fresh change. Preserve existing work and publish the existing requested
bookmark over SSH with the configured identity and signing key.
