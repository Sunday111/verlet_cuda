# Particle performance investigation

Measured on 2026-09-13 against baseline `80584b3`. The target workload is 1–2 million particles.

## Changes

Physics uses a 20-byte `VerletObject`; the 24-byte colour/scale record is stored in a separate `VerletAppearance` buffer. Rendering binds both arrays at the same instance index. Allocation growth and spawning preserve their order, and the application accepts up to two million particles. Total particle storage remains 44 bytes per particle.

The collision solver holds the current object position locally across all nine neighbouring cells, then writes it back before visiting the next origin object. Collision arithmetic and traversal order are preserved. This relies on fixed, unique grid membership during each sweep and disjoint neighbourhoods within each of the nine phases.

## Full application measurements

RTX 4080 SUPER, driver 610.57.04, 64 MiB L2, Clang 22.1.8, CUDA 13.3.73, Release `-O3 -DNDEBUG`, configured `sm_75` target. Both applications were built and launched through YAE with NVIDIA PRIME settings.

Temporary instrumented copies seeded identical physics fixtures matching the headless benchmark and assigned index-dependent colours and scales. Rendering used the default camera, offscreen at 1920×1080 with a fixed simulation clock and no frame-rate cap. The fixtures evolved for 120 frames. Wall-clock intervals between successive `Tick` entries include simulation, rendering and engine overhead; the first 20 intervals were discarded. Results pool 99 intervals from each of two runs with reversed baseline/candidate order. The temporary baseline cap was raised to two million to permit the comparison.

| Particles | Scene | Baseline frame | Optimised frame | Speed-up |
| ---: | --- | ---: | ---: | ---: |
| 1,000,000 | dense | 15.50 ms | 9.15 ms | 1.69× |
| 1,000,000 | sparse | 10.37 ms | 6.78 ms | 1.53× |
| 1,500,000 | dense | 28.52 ms | 15.63 ms | 1.82× |
| 1,500,000 | sparse | 22.14 ms | 12.54 ms | 1.77× |
| 2,000,000 | dense | 69.04 ms | 24.36 ms | 2.83× |
| 2,000,000 | sparse | 61.59 ms | 20.99 ms | 2.93× |

These are medians for evolving fixtures, not a fixed frame-rate guarantee. Collision ordering after atomic grid population is nondeterministic, and work changes as the scene settles. Rendering accesses the appearance buffer as well as physics state, so full-application gains are smaller than the simulation-only gains below.

## Headless kernel measurements

The opt-in [benchmark](readme.md#kernel-benchmark) uses production wrappers and CUDA events. Each run warms its selected workload for two seconds, then measures 50 samples with resets outside the timed region. The final matrix alternated variant order across scenes. These frames restart from the same initial state; they do not model an evolving render loop.

| Particles | Scene | Sweep baseline → optimised | Frame baseline → optimised |
| ---: | --- | ---: | ---: |
| 1,000,000 | sparse | 0.649 → 0.437 ms | 5.349 → 2.999 ms |
| 1,000,000 | dense | 1.608 → 0.923 ms | 14.949 → 6.703 ms |
| 1,000,000 | lattice | 0.327 → 0.125 ms | 3.694 → 1.404 ms |
| 1,000,000 | coincident | 1.025 → 0.427 ms | 9.544 → 4.003 ms |
| 1,500,000 | sparse | 1.964 → 0.730 ms | 17.655 → 5.074 ms |
| 1,500,000 | dense | 2.963 → 1.087 ms | 28.096 → 8.029 ms |
| 1,500,000 | lattice | 0.587 → 0.170 ms | 6.848 → 2.060 ms |
| 1,500,000 | coincident | 1.403 → 0.532 ms | 13.307 → 5.160 ms |
| 2,000,000 | sparse | 4.902 → 1.322 ms | 49.161 → 11.136 ms |
| 2,000,000 | dense | 6.326 → 1.803 ms | 61.716 → 14.443 ms |
| 2,000,000 | lattice | 2.772 → 0.649 ms | 26.791 → 5.898 ms |
| 2,000,000 | coincident | 2.568 → 0.736 ms | 24.456 → 6.844 ms |

## Working set

The grid contains 1,922 × 1,042 four-byte cells. Approximate particle-plus-grid storage touched by physics is:

| Particles | Original layout | Compact physics layout |
| ---: | ---: | ---: |
| 1,000,000 | 49.60 MiB | 26.71 MiB |
| 1,500,000 | 70.58 MiB | 36.25 MiB |
| 2,000,000 | 91.56 MiB | 45.79 MiB |

The original layout exceeds the GPU’s 64 MiB L2 capacity between one and one-and-a-half million particles. The compact physics layout remains below it at two million. Rendering still reads the separate appearance data; this is a smaller physics working set, not a reduction in total particle storage.

## Validation

- The 121 pair regression cases and a six-particle multi-cell regression pass on the GPU. The latter exercises two origins and four neighbouring cells against an independent double-precision oracle; removing the production cache flush makes it fail.
- 10,000 independent pairs pass a double-precision oracle with maximum absolute error below `8.5e-8`.
- All 12 deterministic sweep outputs (three counts × four scenes) match baseline positions byte for byte.
- All 12 evolving simulations complete 120 frames with finite positions inside the two-unit integration boundary after every frame.
- A one-million-particle lattice capture at frame 30 matches the baseline byte for byte, including varying colours and scales.
- Real application allocation/upload checks preserve every physics and appearance field through empty upload, initial allocation, no-growth upload, reallocation, filling to two million, and rejection beyond the cap.
- The production application builds through YAE; the CUDA–Vulkan path is exercised by the full-application comparisons.

The benchmark and long validation runs are opt-in. They are not part of the default build or run.

## Other experiments

Fixed grid dimensions removed runtime division and improved smaller workloads, but offered little in the target range. Collision blocks of 64, 128, 256 and 512 threads gave mixed results, including two-million-particle regressions, so the existing 1,024-thread size is retained. A local `__fdividef` substitution reduced instruction count but did not materially improve the dense two-million-particle frame; the original division remains.

Further work can investigate separate position/index arrays or contiguous per-cell ranges. Measure their rebuild and rendering costs before replacing the linked grid; the current results do not establish that those changes will help.
