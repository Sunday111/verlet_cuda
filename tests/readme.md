# Collision regression

`collisions.cu` runs 121 cases against the production collision routine on the GPU without a window. It checks coincident
and deeply penetrating pairs, underflowing and subnormal squared distances, both traversal orders, horizontal, vertical and
diagonal separation, conservation of the pair midpoint, touching or separated particles, and self-collision rejection.

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
