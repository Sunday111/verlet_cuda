# Clone configure and build

For this project you will need CUDA toolkit installed in your system.


# Clone

This project uses my scripts to clone dependencies and generate CMake files. It lives in `./yae` submodule. So you have to clone repo and update submodule.

```bash
git clone https://github.com/Sunday111/verlet_cuda
cd verlet_cuda
git submodule update --init
```

# Dependencies and cmake files

Now you can invoke the script with python

```bash
python ./yae/scripts/make_project_files.py --project_dir=.
```

# Generate project files

```bash
cmake -S . -B ./build
```

# Build

```bash
cmake --build ./build --config Release --target verlet_cuda
```

It builds only the simulation project itself in release configuration without tests, benchmarks and examples from dependencies.

With cmake generators that do not use parallel building by default you might want to build in parallel explicitly:

```bash
cmake --build ./build --config Release --target verlet_cuda --parallel
```

Or open generated project files in editor of choice.
