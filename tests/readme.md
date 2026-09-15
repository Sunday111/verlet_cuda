# Collision regression

`collisions.cu` runs 121 cases against the production collision routine on the GPU without a window. It checks coincident
and deeply penetrating pairs, underflowing and subnormal squared distances, both traversal orders, horizontal, vertical and
diagonal separation, conservation of the pair midpoint, touching or separated particles, and self-collision rejection.
A six-particle regression exercises two origins in one cell and four neighbouring cells through the production cell solver,
comparing every final position with an independent double-precision reference to check cached-position propagation.
A six-particle integration regression checks separate previous-position storage, nonzero velocities, gravity and world bounds.

With Clang, CUDA installed in `/opt/cuda`, and the dependencies fetched:

```sh
clang++ -O3 -DNDEBUG -std=c++23 --cuda-gpu-arch=sm_75 --cuda-include-ptx=sm_75 \
  --cuda-path=/opt/cuda -Wno-unknown-cuda-version \
  -I/opt/cuda/include/cccl -I src/verlet_cuda/code/private \
  -I"$YAE_CLONED_REPOSITORIES_DIR/Sunday111/edt/main/modules/edt/code/public" \
  -I"$YAE_CLONED_REPOSITORIES_DIR/fmtlib/fmt/12.2.0/include" \
  tests/collisions.cu -L/opt/cuda/lib64 -Wl,-rpath,/opt/cuda/lib64 -lcudart -o /tmp/collision-test
/tmp/collision-test
```

Select the CUDA architecture matching the application build. The test requires a CUDA-capable GPU.

# Kernel benchmark

`benchmark.cu` measures the production kernel wrappers without Vulkan or a window. Build it with the collision regression
command above, replacing `tests/collisions.cu` with `tests/benchmark.cu` and `/tmp/collision-test` with `/tmp/kernel-benchmark`.
Use the same compiler, optimisation flags and CUDA architecture for baseline and candidate builds.

For a 3840×2160 physics world, add `-DVERLET_WORLD_WIDTH=3840 -DVERLET_WORLD_HEIGHT=2160` to the compiler command.
The default world is 1920×1040. These definitions control simulation bounds and grid size, not a rendering framebuffer.
Use the same definitions for every translation unit in an executable. The standalone benchmark and regressions each
compile as one translation unit. Measurements exclude rendering, rendering-buffer copies and CUDA–Vulkan synchronisation.

```sh
/tmp/kernel-benchmark 1000000 packed evolve 100 > /tmp/packed-1m.csv
/tmp/kernel-benchmark 1500000 packed evolve 100 > /tmp/packed-1.5m.csv
/tmp/kernel-benchmark 2000000 packed evolve 100 > /tmp/packed-2m.csv
```

The arguments are particle count (`100000`, `1000000`, `1500000`, `2000000`, `3000000`, or `4000000`), scene (`sparse`,
`dense`, `lattice`, `coincident`, `packed`, or `burst`), mode (`grid`, `sweep`, `frame`, or `evolve`), optional sample count
(1–100000, default 50), and optional collision backend (`direct` or `cached`). The default backend is `cached` at three
million particles and above, matching the application, and `direct` below that threshold.
Performance decisions target 1–2 million particles; the 100k case is available for smaller checks. Run 1m, 1.5m and 2m with
`packed` in `evolve` mode before adopting a kernel optimisation. The full population is uploaded before warmup; no measured
mode includes spawning. `grid`, `sweep` and `frame` provide reset-based diagnostics; sparse cases are optional.
Repeat baseline and candidate runs in alternating order to expose clock and temperature drift; compare the distribution
of samples, not just the fastest result. GPU identity goes to stderr and individual CUDA-event measurements in milliseconds
go to stdout as CSV.

The packed scene uses staggered rows with 1.01-unit nearest-neighbour spacing, centred horizontally and aligned with the
bottom simulation boundary. It uses the configured world's usable width. Bounds and neighbouring-pair separation are
checked before upload, so a population that does not fit without initial overlap fails before measurement.

The `burst` scene uses the production `BurstLayout::Generate` with packing and shuffled storage enabled:
1.001-diameter staggered spacing and the emitter's fixed shuffle seed. Use a 3840×2160 world for four million particles.
This isolates the emitter's geometry and storage order in the kernel harness; use the application benchmark below to
include the actual emitter and rendering loop.

```sh
/tmp/kernel-benchmark 4000000 burst evolve 100 > /tmp/burst-4m.csv
```

The other scenes provide stress cases. The seeded sparse scene distributes particles across a 1900×1010 region;
its density rises with particle count. The dense scene uses 1.5 particles per square world unit and starts with overlap;
it cannot fit without expansion or compression. The regular lattice uses `ceil(sqrt(locations * 1.83))` columns and enough
rows to hold every location. Its spacing is `min(1.1, 1900 / columns, 1010 / rows)` world units so the layout fits inside the
world. At two million particles the lattice spacing is below one unit and adjacent particles overlap. Coincident particles
occur in pairs at the same lattice sites, using half as many locations; their sites remain 1.1 units apart up to two million
particles. All scenes begin at rest and fit inside the simulation bounds. Initial grid lists have a fixed traversal order.

In `grid`, `sweep` and `frame` modes, each sample restores the same objects, previous positions and grid before its start event,
excluding reset copies from timing. `grid` measures clearing and population alone. A sweep measures the nine collision
launches; the cached backend also includes gathering and restoring the collision state. A frame measures eight simulation
substeps, including grid clearing, population, collision sweeps and integration.
Frame grid population builds ascending particle-index lists with ordered atomic insertion. This insertion cost is included
in `grid`, `frame` and `evolve`, but not in `sweep`. Frames reset between samples rather than advancing an evolving scene.
The selected workload warms up for at least two seconds before measurement. CUDA calls are checked and final positions are
checked for finiteness and world bounds. A final state hash covers positions, previous positions and canonical cell links,
with state restoration and hashing outside the measured interval. These checks do not replace the collision regression or
longer simulation validation.

`evolve` creates the full particle population before measurement. After the reset-based warmup, it restores the initial
state once, advances 120 untimed frames, and measures consecutive simulation frames without resetting or spawning.
The sample count controls how many subsequent frames are measured. This mode excludes allocation and spawning costs
and includes changes in collision work as the existing particles move. Compare the same frame interval between builds.

# Determinism regression

Build `tests/determinism.cu` using the collision regression command, with output `/tmp/determinism-test`.
Run `/tmp/determinism-test` on a CUDA-capable GPU. It compares 24 grid populations with independently built CPU lists,
including stale links, world-edge cells, partial blocks, varied launch sizes and 4,096 particles in a single cell.
Seven bitwise simulation replays then check a reference whose grid lists are built on the CPU. Each replay advances
4,096 particles through 32 production substeps, including coincident particles and nonzero velocity. The comparison includes
positions, previous positions and cell links. Tests are opt-in.

Grid population builds cell lists in ascending particle-index order. Each atomic minimum keeps the smaller index
at the current link and continues inserting the larger index after it. The resulting list order does not depend on scheduling.
Acquire/release ordering publishes initialised links before another thread follows them. The nine sequential collision
passes operate on disjoint 3-by-3 cell neighbourhoods within each pass; cell membership remains fixed during the sweep.
Together these provide a fixed arithmetic order for identical initial state and per-frame inputs on the same GPU and build.
Each pass is a template specialisation with compile-time grid dimensions and offset. Bitwise agreement across
architectures, compiler options or different input timing is not promised.

Ordered insertion requires no scratch allocation, additional kernel launch or separate grid scan. Its worst-case
work is O(k squared) for a cell containing k particles, with contention on the cell's links. Use the 1m/1.5m/2m benchmark
matrix above to assess performance after all objects have spawned. Traversal order also changes subsequent collision work,
so compare the same settling period and measured frame interval.

# Spatial collision cache

At three million particles and above, the application groups working particles by grid cell, keeping their original indices
in a separate array. It retains this working order for eight frames. Each substep still rebuilds the grid, sorts cell lists
by original particle index, and executes the same nine collision passes with the same arithmetic. Coincident-particle
separation also uses the original index. A block reduction records the last occupied cell so collision passes can skip
empty trailing grid cells without omitting moving particles.

Previous positions use the same working order. Render positions return to their original shuffled slots every frame;
canonical history is restored before regrouping or spawning. Regrouping is an allocation-free gather into reusable buffers.
The additional storage is approximately 24 bytes per reserved particle (about 92 MiB for four million particles).
The direct path remains available for smaller workloads and benchmark comparisons.

```sh
/tmp/kernel-benchmark 4000000 burst evolve 100 direct > /tmp/burst-direct.csv
/tmp/kernel-benchmark 4000000 burst evolve 100 cached > /tmp/burst-cached.csv
```

See [performance measurements](performance.md) for the measured workload and limitations.

# Application world size and capacity

The application defaults to a 1920×1040 physics world and a maximum of 4,000,000 particles.
For larger populations, configure the world and capacity after the initial YAE configuration:

```sh
cmake -S . -B build -DVERLET_WORLD_WIDTH=3840 -DVERLET_WORLD_HEIGHT=2160 -DVERLET_MAX_OBJECTS=4000000
yae build verlet_cuda
```

These positive-integer cache settings apply to all application translation units, including CUDA.
The capacity is an allocation limit, not a guarantee that particles fit without overlap in the default world.
Use the larger world above for four million particles at the production radius of 0.5.
Restore the defaults by setting width 1920, height 1040 and maximum objects 4000000.

# Burst emitter and application benchmark

In the application, open **Emitters → New Burst**, then enable the emitter. It queues the entire remaining
particle budget in one activation, at rest, and uploads it together on the next frame. It does not emit batches
over successive frames. **Rearm** permits another burst if capacity remains; cloning creates a disabled, rearmed copy.

Both options are enabled by default:

- **Shuffle storage** permutes the entire population with a fixed seed before insertion, reducing spatial locality.
  Disabling it retains row order. Shuffling changes storage order, not the initial set of positions.
- **Packing** uses dense staggered rows at 1.001 particle diameters, starting at the bottom world margin.
  Disabling it distributes positions over the whole world on an even rectangular grid.

A packed burst that cannot fit spawns nothing and disables the emitter with an explanatory message.
The layout concerns the new burst; it does not prevent overlap with particles already in the simulation.
Packed, shuffled storage is a demanding controlled workload, not a guarantee of the worst possible simulation state.

The opt-in application benchmark invokes this same `BurstEmitter::Tick` and uses the ordinary simulation/rendering loop.
After the entire burst is generated and uploaded, it advances 120 untimed frames by default, then measures 300 frames.
Generation, upload, and settling are outside timing. Both GPU APIs are drained at the measurement boundaries;
the reported duration is completed-batch wall time divided by the frame count, including simulation and rendering.
The application FPS cap is disabled for this measurement. Swapchain/compositor pacing can still affect visible runs;
these are throughput measurements, not physical scanout timestamps.

```sh
python3 tests/burst_benchmark.py --output /tmp/verlet-burst
```

The default runs 2m, 3m and 4m particles with both options enabled, using **visible windows** that close automatically.
Visible runs use the window size assigned by the desktop; output records the actual framebuffer dimensions.
Configure the larger world above first. `--particles`, `--warmup`, and `--samples` customise the run;
`--no-shuffle-storage` and `--no-packing` disable the corresponding emitter options.
For headless validation, explicitly use `--presentation offscreen`; keep that result separate from visible timings.
Logs, configurations, per-run CSVs and `summary.json` are saved under the required output directory.
Offscreen runs also save final captures.

The CPU-only layout regression checks bounds, non-overlap, rejected oversized packing, and identical geometry across
ordered and shuffled storage. Run it explicitly:

```sh
clang++ -O2 -std=c++23 -DVERLET_WORLD_WIDTH=3840 -DVERLET_WORLD_HEIGHT=2160 \
  -I src/verlet_cuda/code/private \
  -I "$YAE_CLONED_REPOSITORIES_DIR/Sunday111/edt/main/modules/edt/code/public" \
  tests/burst_layout.cpp -o /tmp/burst-layout-test
/tmp/burst-layout-test
```
