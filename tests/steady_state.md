# Steady-state performance

Baseline: `d300a36`. Measured on 2026-09-14 with an RTX 4080 SUPER, driver 610.57.04,
Clang 22.1.8, CUDA 13.3, Release `-O3 -DNDEBUG`, and CUDA architecture `sm_75`.
The target workload is one to two million particles already present in the simulation.

## Storage

Collision records contain the current position and next-cell index (12 bytes).
Previous positions occupy a separate eight-byte array. Both simulation arrays use
CUDA allocations. After all eight substeps finish, one device-to-device copy publishes
the completed collision records into the Vulkan-shared vertex buffer. Rendering uses
the same particle indices, colours, scales and triangle order.

The collision equations, arithmetic order, cell traversal, nine collision phases,
integration step and damping remain unchanged. The draw reads the completed current
frame; the snapshot introduces no frame of latency.

The rendering snapshot adds 12 bytes per particle: total particle storage rises from
44 to 56 bytes, excluding allocation capacity rounding. At two million particles this
adds 24,000,000 bytes (22.89 MiB) and copies 24,000,000 bytes once per rendered frame.
The allocation growth policy is unchanged.

Separating previous positions while retaining Vulkan-shared simulation storage yielded
only about 1–4% improvement in the full application. CUDA-owned simulation storage,
including the copy cost, produced the larger gains below. This is a measured difference
between the storage paths, not a hardware-level diagnosis of the shared-buffer cost.

## Full application

Temporary instrumented builds created the entire seeded population before the first
simulation tick, using the dense and sparse distributions in `benchmark.cu`. They used
index-dependent colours, the default camera, offscreen 1920×1080 rendering, a fixed
simulation clock and no frame-rate cap. No particles were spawned during measurement.
Both versions were built through YAE and the preserved executables ran with NVIDIA
PRIME settings. Each run advanced 240 frames. The table pools 120 successive Tick-entry
intervals from each of two runs, reversing baseline/candidate order in the second run.
Intervals before tick 120 were excluded. These are full frame intervals, including
simulation, the snapshot copy, rendering and engine work.

| Particles | Scene | Baseline frame | Candidate frame | Speed-up |
| ---: | --- | ---: | ---: | ---: |
| 1,000,000 | dense | 8.98 ms | 4.44 ms | 2.02× |
| 1,000,000 | sparse | 8.23 ms | 4.08 ms | 2.02× |
| 1,500,000 | dense | 15.87 ms | 7.57 ms | 2.10× |
| 1,500,000 | sparse | 14.13 ms | 6.31 ms | 2.24× |
| 2,000,000 | dense | 24.02 ms | 10.71 ms | 2.24× |
| 2,000,000 | sparse | 22.96 ms | 9.86 ms | 2.33× |

The existing atomic grid builder is nondeterministic, so evolving simulations do not
have a general bitwise trajectory guarantee even within the same build. These timings
are fixture measurements on this GPU, not a universal frame-rate promise.

## Headless evolving simulation

The opt-in benchmark supports `evolve`: it restores the full initial population once,
advances 120 untimed frames, then measures consecutive frames without resets. For example:

```sh
/tmp/kernel-benchmark 2000000 dense evolve 120
```

Build instructions are in [the test guide](readme.md). The following medians compare
CUDA-owned storage in both versions to isolate the smaller collision record. The
candidate prototype used a device pointer for previous positions; the final production
wrapper passes that pointer explicitly. A separate final-wrapper dense two-million run
measured 9.03 ms. Headless timings exclude rendering and the rendering snapshot copy.

| Particles | Scene | Original record | Smaller record |
| ---: | --- | ---: | ---: |
| 2,000,000 | dense | 11.573 ms | 9.024 ms |
| 2,000,000 | sparse | 10.902 ms | 8.318 ms |
| 2,000,000 | lattice | 6.968 ms | 5.488 ms |
| 2,000,000 | coincident | 35.328 ms | 34.435 ms |
| 1,000,000 | dense | 3.660 ms | 3.502 ms |
| 1,000,000 | sparse | 3.183 ms | 3.038 ms |
| 1,000,000 | lattice | 1.425 ms | 1.029 ms |
| 1,000,000 | coincident | 10.023 ms | 9.872 ms |
| 1,500,000 | dense | 6.470 ms | 6.220 ms |
| 1,500,000 | sparse | 5.154 ms | 4.940 ms |
| 1,500,000 | lattice | 2.328 ms | 1.441 ms |
| 1,500,000 | coincident | 19.835 ms | 19.642 ms |

## Validation

- All twelve fixed-grid sweep comparisons (three counts × four scenes) preserve every
  particle position and linked-cell index byte for byte.
- The 121 collision cases, multicell regression and new separate-array integration
  regression pass on the GPU. CUDA Compute Sanitizer reports zero memory errors.
- Ten thousand seeded particles with nonzero velocities match baseline current and
  previous positions byte for byte after eight integration steps.
- All twelve headless evolving workloads complete the 120-frame prelude and 120 measured
  frames with finite final positions inside the world bounds.
- A one-million-particle lattice capture at frame 30, with index-dependent colours,
  matches baseline byte for byte through the real CUDA–Vulkan path.
- Application storage checks preserve current positions, previous positions, cell links,
  colours and scales through empty upload, initial allocation, no-growth upload, multiple
  growth boundaries, filling to two million, and rejection beyond the cap.
- The production application builds through YAE. Benchmark and long simulation checks
  remain opt-in.
