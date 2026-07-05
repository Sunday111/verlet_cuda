# Clone, Configure And Build

For this project you will need CUDA toolkit installed in your system.


# Clone

This project uses `yae` to clone dependencies, generate CMake files, configure CMake, build, format and run.

```bash
git clone https://github.com/Sunday111/yae
git clone https://github.com/Sunday111/verlet_cuda
cd verlet_cuda
```

You can invoke `yae` directly from its checkout:

```bash
../yae/yae configure
../yae/yae build
../yae/yae run
```

Or install a system symlink and call it as `yae`:

```bash
sudo ln -sf "$HOME/github/Sunday111/yae/yae" /usr/bin/yae
```

Compiler, generator, build directory, build targets and the default run target live in `yae_project.json` under
`default_configuration`.

Machine-specific overrides can go in ignored `local-config.json` next to `yae_project.json`.

# Configure and build

```bash
yae build
```

# Run

```bash
yae run
```
