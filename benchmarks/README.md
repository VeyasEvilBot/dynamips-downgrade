# Guest memory benchmark

Run `mise run test:memory` for correctness and `mise run bench:memory` for
optimized timings. The benchmark task selects release mode and system packages.
For repeatable CPU affinity on Linux, run the built executable with
`taskset -c 2 build/linux/x86_64/release/memory-bench`, choosing an available CPU.

The benchmark performs big-endian 64-bit RAM loads, uses precomputed address
traces, checks every path against raw RAM, and reports the median of five samples
of 1,048,576 accesses. Paths are interleaved with rotating order for each sample.
Setup and mapping are outside the timed sections. The
new direct path permits inlining; callback paths use an opaque function pointer.
Raw RAM is a lower-bound reference without address translation or permissions.

The legacy fixture uses the real `mips64_mts64_init`, memory-operation vector,
and RAM device lookup from Dynamips, including cache refills. It does not model
the legacy algorithm in benchmark code. Its extra memory-operation dispatch and
register write are part of that callback contract. These results do not compare
against the legacy inline JIT memory path or establish whole-emulator speedups.

The workloads cover a hot 16-page working set, L1 conflicts, large page strides,
legacy-cache conflicts, and random accesses over 4,096 pages. The folded L2 index
reduces aligned-stride conflicts, but any finite direct-mapped cache can still
thrash. The correctness test includes addresses that collide in both new levels.

`jit.memory` currently owns an explicit page map for one execution context.
RAM occupies physical `[0, ram.size())`; MMIO pages cannot overlap it. RAM and
device objects are borrowed and must remain stable until unmapped. Mapping and
callback changes require stopped execution. Every mapping change invalidates
affected entries in all access caches. No allocation occurs on RAM hits, cache
refills, or mapping changes within the startup capacity.

This is not an architectural MIPS CP0 TLB implementation. ASIDs, segment rules,
guest TLB instructions, precise architectural exception delivery, and inline JIT
cache probing remain separate work. MMIO callbacks receive the requested byte
order and exchange numeric values once per access; scalar RAM operations support
explicit big/little endianness and
natural alignment for 1, 2, 4, and 8 bytes.


## IOS boot timing

Build both standalone binaries in release mode with the AMD64 JIT selected
(`--dynamips_arch=amd64`), then compare the same extracted C7200 IOS ELF:

```sh
python3 benchmarks/boot.py build/linux/x86_64/release/dynamips \
  --core legacy --image /path/to/c7200.image --label legacy --cpu 2
python3 benchmarks/boot.py build/linux/x86_64/release/c7200-boot \
  --image /path/to/c7200.image --label new --cpu 2
```

Choose an available CPU. The harness records three warm-file-cache process
starts, measures launch to the first IOS banner, and writes console logs and
JSON under `build/boot-bench`. Both machines use NPE-400 and 256 MiB RAM. The
legacy console uses TCP because its terminal transport does not emit guest
output through a redirected terminal. Legacy execution stops at the banner;
the new core exits at its documented unsupported-device checkpoint. These
measurements do not compare CLI readiness or networking.

For a separate CPU sample, pass `--runs 1 --profiler /path/to/libprofiler.so`
to the new-core command and keep release symbols/frame pointers in that
binary. Dependencies of the profiler must be on the library search path.
Google perftools may append a PID to the `.prof` filename. Analyze the result
with `go tool pprof -top -cum /path/to/binary /path/to/profile`. Sampled runs
include profiler overhead and must not be used for before/after timings.

The JSON also reports peak child RSS (KiB on Linux) across the run group.
It includes the full process lifetime, including teardown after the banner.
