# Read-only collision links

Baseline: `d7ce84a`. Measured on 2026-09-14 with an RTX 4080 SUPER, driver 610.57.04,
Clang 22.1.8, CUDA 13.3, Release `-O3 -DNDEBUG`, and CUDA architecture `sm_75`.

## Change and correctness constraint

The collision solver uses `__ldg` for grid heads and next-object indices. These fields
are built by `PopulateGrid` and remain unchanged throughout all nine collision phases.
Position reads and writes, collision arithmetic, traversal order, phase order, integration,
allocation and rendering are unchanged. No buffers or kernel launches are added.

[NVIDIA's programming guide](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#read-only-data-cache-load-function)
documents the read-only data cache load. Every address passed to `__ldg` must refer to
global data that remains read-only for the entire kernel. The regression fixtures therefore
initialise grid heads and links before launching the collision kernels, in global storage.
Creating those links inside the kernel or passing a thread-local grid would violate this
requirement. Generated PTX contains `ld.global.nc` for these loads.

The atomic grid builder already has nondeterministic insertion order. This change preserves
the calculation for a given grid order; neither version guarantees identical complete
trajectories across runs.

## Headless measurements

The opt-in [kernel benchmark](readme.md#kernel-benchmark) runs without Vulkan or a window.
Both builds used identical compiler flags. Each workload warmed up for two seconds and
collected 50 CUDA-event samples. The complete matrix covers three counts, four scenes and
`sweep`, `frame` and `evolve`, repeated with variant order reversed. Evolving runs begin
with all particles present, advance 120 untimed frames, then measure the next 50 frames.
These medians pool both repeats. They exclude rendering and the rendering snapshot copy.

| Particles | Scene | Baseline evolving frame | Candidate evolving frame | Speed-up |
| ---: | --- | ---: | ---: | ---: |
| 1,000,000 | dense | 3.250 ms | 2.871 ms | 1.13× |
| 1,000,000 | sparse | 2.893 ms | 2.738 ms | 1.06× |
| 1,000,000 | lattice | 1.029 ms | 0.965 ms | 1.07× |
| 1,000,000 | coincident | 9.855 ms | 8.385 ms | 1.18× |
| 1,500,000 | dense | 6.034 ms | 5.456 ms | 1.11× |
| 1,500,000 | sparse | 4.627 ms | 4.050 ms | 1.14× |
| 1,500,000 | lattice | 1.355 ms | 1.281 ms | 1.06× |
| 1,500,000 | coincident | 19.598 ms | 16.863 ms | 1.16× |
| 2,000,000 | dense | 9.178 ms | 8.637 ms | 1.06× |
| 2,000,000 | sparse | 7.477 ms | 6.958 ms | 1.07× |
| 2,000,000 | lattice | 4.740 ms | 4.579 ms | 1.04× |
| 2,000,000 | coincident | 34.334 ms | 30.068 ms | 1.14× |

## Full application

Temporary instrumented executables seeded the benchmark's dense and sparse scenes before
simulation, with index-dependent colours. They rendered offscreen at 1920×1080 with a fixed
clock, the default camera and no frame-rate cap. Each run advanced 240 frames; intervals
between Tick entries 120–239 measure simulation, the snapshot copy, rendering and engine
work. The table pools 120 intervals from each of two runs with reversed variant order.
No spawning or allocation occurs during measurement.

| Particles | Scene | Baseline frame | Candidate frame | Speed-up |
| ---: | --- | ---: | ---: | ---: |
| 1,000,000 | dense | 4.526 ms | 4.108 ms | 1.10× |
| 1,000,000 | sparse | 4.049 ms | 3.854 ms | 1.05× |
| 1,500,000 | dense | 7.565 ms | 7.071 ms | 1.07× |
| 1,500,000 | sparse | 6.253 ms | 5.782 ms | 1.08× |
| 2,000,000 | dense | 10.697 ms | 10.030 ms | 1.07× |
| 2,000,000 | sparse | 9.810 ms | 9.263 ms | 1.06× |

These are fixture measurements on this GPU, not universal frame-rate guarantees.

## Validation

- All twelve fixed-grid sweep cases match the baseline byte for byte, including every
  position and linked-cell index, in both repetitions. Temporary benchmark instrumentation
  downloaded and compared the complete particle arrays after each sweep workload.
- Both versions pass all 121 pair cases, the multicell oracle and the integration regression.
  CUDA Compute Sanitizer reports zero memory errors for the final regression executable.
- All evolving workloads complete their 120-frame prelude and 50 measured frames in both
  repetitions, with finite final positions inside the world bounds.
- The full CUDA–Vulkan application produces a byte-identical one-million-particle lattice
  capture at frame 30, including index-dependent colours.
- The production application builds through YAE in a fresh build directory. Tests and
  benchmarks remain opt-in.

## Rejected candidates

Collision blocks of 128, 256 and 512 threads gave inconsistent results at two million
particles: smaller blocks helped dense and coincident scenes but slowed sparse simulations.
Caching all nine neighbouring cell heads in registers gave smaller gains. CUDA graph replay
was approximately neutral in the target evolving workloads. None of these changes is included.
