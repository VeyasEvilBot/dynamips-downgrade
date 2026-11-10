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
guest TLB instructions, precise CPU exception delivery, and JIT load/store IR
integration remain separate work. MMIO callbacks receive the requested byte order
and exchange numeric values once per access; scalar RAM operations support explicit big/little endianness and
natural alignment for 1, 2, 4, and 8 bytes.
