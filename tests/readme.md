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

```sh
/tmp/kernel-benchmark 1000000 dense sweep 50 > /tmp/dense-sweep.csv
/tmp/kernel-benchmark 2000000 coincident frame 50 > /tmp/coincident-frame.csv
/tmp/kernel-benchmark 2000000 dense evolve 120 > /tmp/dense-evolving.csv
```

The arguments are particle count (`100000`, `1000000`, `1500000`, or `2000000`), scene (`sparse`, `dense`, `lattice`, or
`coincident`), mode (`sweep`, `frame`, or `evolve`), and optional sample count (1–100000, default 50). Performance decisions target
1–2 million particles; the 100k case is available for smaller checks. Run 1m, 1.5m and 2m with all four scenes and all three modes
before adopting a kernel optimisation. Repeat baseline and candidate runs in alternating order to expose clock and
temperature drift; compare the distribution of samples, not just the fastest result. GPU identity goes to stderr and individual CUDA-event
measurements in milliseconds go to stdout as CSV. See [the performance investigation](performance.md) for candidate changes
and generated-code observations.

The seeded sparse scene distributes particles across most of the world; its density rises with particle count. The dense
scene uses 1.5 particles per square world unit. The regular lattice uses `ceil(sqrt(locations * 1.83))` columns and enough
rows to hold every location. Its spacing is `min(1.1, 1900 / columns, 1010 / rows)` world units so the layout fits inside the
world. At two million particles the lattice spacing is below one unit and adjacent particles overlap. Coincident particles
occur in pairs at the same lattice sites, using half as many locations; their sites remain 1.1 units apart at every supported
count. All scenes begin at rest and fit inside the simulation bounds. Initial grid lists have a fixed traversal order.

In `sweep` and `frame` modes, each sample restores the same objects, previous positions and grid before its start event,
excluding reset copies from timing. A sweep measures the nine collision launches. A frame measures eight simulation substeps, including grid clearing, population, collision
sweeps and integration. Frame grid population uses atomics, so its traversal order and collision results are not
deterministic even though the initial scene is fixed. Frames reset between samples rather than advancing an evolving scene.
The selected workload warms up for at least two seconds before measurement. CUDA calls are checked and final positions are
checked for finiteness and world bounds; these checks do not replace the collision regression or longer simulation validation.

`evolve` creates the full particle population before measurement. After the reset-based warmup, it restores the initial
state once, advances 120 untimed frames, and measures consecutive simulation frames without resetting or spawning.
The sample count controls how many subsequent frames are measured. This mode excludes allocation and spawning costs
and includes changes in collision work as the existing particles move. Compare the same frame interval between builds.
See [the steady-state investigation](steady_state.md) for measurements with separate previous-position storage.
