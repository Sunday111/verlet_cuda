# Packed, shuffled burst performance

Measured on 2026-09-15 against application revision `4bfee7b`.

## Workload

- NVIDIA GeForce RTX 4080 SUPER, driver 610.57.04; clocks were not locked.
- Clang 22.1.8, Release / `-O3 -DNDEBUG`, C++23, CUDA target `sm_75`.
- Four million particles in a 3840×2160 physics world, radius 0.5.
- Production burst layout with packing and shuffled storage enabled: 1.001-diameter spacing, seed 12345.
- Fixed 1/60-second simulation step with eight substeps. Allocation, spawning and the first 120 frames are untimed.

The kernel harness uses the emitter's layout generator. The application measurements invoke `BurstEmitter::Tick` and
include the ordinary simulation, snapshot copy, CUDA–Vulkan synchronisation and rendering loop.

## Kernel improvements

Mean CUDA-event times for 100 consecutive frames after settling. The first spatial-buffer measurement precedes the three
subsequent optimisation iterations. Each row includes all preceding changes. The final result averages two runs.

| Implementation | Kernel ms/frame | Reduction from original |
| --- | ---: | ---: |
| Original solver | 36.64 | — |
| Spatial collision buffer, rebuilt each substep | 21.16 | 42.2% |
| Iteration 1: reuse working order across eight substeps | 20.73 | 43.4% |
| Iteration 2: store previous positions in working order | 18.40 | 49.8% |
| Iteration 3: reuse across eight frames, bound occupied cells, restore history only when needed | 14.49 | 60.5% |

The original particle indices still determine every cell-list traversal and coincident-particle tie break. All these
measurements produced the same canonical final-state hash, `b338dde62d48c939`, covering positions, previous positions and
cell links. Working-buffer allocation order does not determine arithmetic order.

## Full application

Offscreen framebuffer: 1920×1080. Each baseline/final pair used the same frame interval and generated an identical klvk
capture after timing. The three 300-frame pairs ran in alternating baseline/final order. These are completed-batch
throughput measurements, not physical display presentation measurements or per-frame latency percentiles.

| Measured interval | Original ms/frame | Optimised ms/frame | Time reduction | Optimised throughput |
| --- | ---: | ---: | ---: | ---: |
| 300 frames, mean of three runs | 39.574 | 18.095 | 54.3% | 55.3 FPS |
| 2,000 frames | 40.248 | 17.916 | 55.5% | 55.8 FPS |

The three optimised 300-frame means ranged from 18.090 to 18.101 ms. The 60 FPS target requires 16.667 ms including rendering;
these measurements do **not** establish stable 60 FPS. The kernel-only result must not be used as the application frame rate.

All three 300-frame captures matched each other and their baselines byte for byte, as did the 2,000-frame baseline/final
pair. Their respective SHA-256 values are:

- Frame 421: `05064f048b9aa661320ab7ebd3b2e7af8ae77c290b4d2ff659541d5286734adb`
- Frame 2121: `319b9df9f5e6d69f57a3d70e68ea23be645dabd1d294efbf0f8a918d5623f61a`

## Reproduce

Build the kernel benchmark using the [test instructions](readme.md#kernel-benchmark), with the larger world definitions.
Both backends are available in the same executable for comparisons:

```sh
/tmp/kernel-benchmark 4000000 burst evolve 100 direct > /tmp/burst-direct.csv
/tmp/kernel-benchmark 4000000 burst evolve 100 cached > /tmp/burst-cached.csv
```

Configure the application world as described in the [application instructions](readme.md#application-world-size-and-capacity),
then run the opt-in full application benchmark:

```sh
python3 tests/burst_benchmark.py --particles 4000000 --presentation offscreen --samples 300 --output /tmp/burst-300
python3 tests/burst_benchmark.py --particles 4000000 --presentation offscreen --samples 2000 --output /tmp/burst-2000
```

Repeat baseline and candidate runs in alternating order, and compare their PPM captures with `cmp`. Keep framebuffer size,
world dimensions, particle count, compiler flags, shuffle settings, warmup and sample count identical within each pair.

## Validation

The collision regression passed all 121 pair cases plus its multicell and integration checks. The existing determinism
regression passed 24 GPU/CPU grid comparisons and seven bitwise replays. The CPU burst-layout regression also passed.

Baseline and final kernel states matched for packed populations of one, one-and-a-half, two and four million particles,
and shuffled bursts of three and four million particles. Forced-cache comparisons with 100,000 coincident particles
also matched in both frame and sweep modes. The application uses the direct backend below three million particles.

## Further optimisation attempts

The next measurements start from the spatial-cache implementation above. The workload, compiler flags and fixed frame
intervals remain the same. Each retained approach has its own commit.

### Fused integration and grid population

The first substep builds the grid normally. Each subsequent substep integrates positions while building its grid, and a
final integration follows the last collision sweep. This preserves all eight integrations and their arithmetic order,
while removing seven separate integration launches and repeated position reads.

Two alternating kernel comparisons measured 14.538 ms for the starting implementation and 14.324 ms for fusion, a 1.5%
reduction. Both produced `b338dde62d48c939`. A fresh 300-frame application comparison measured 18.147 ms versus 17.790 ms,
and its klvk captures matched byte for byte. These are mean throughput results; they do not establish stable 60 FPS.
