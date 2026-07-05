#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
build_dir="${1:-}"
if [[ -z "${build_dir}" ]]; then
    build_dir="${project_dir}/build"
fi

export CCACHE_DIR="${CCACHE_DIR:-"${project_dir}/.cache/ccache"}"
mkdir -p "${CCACHE_DIR}"

python "${project_dir}/yae/scripts/make_project_files.py" --project_dir="${project_dir}"

cmake \
    -S "${project_dir}" \
    -B "${build_dir}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
    -DCMAKE_C_COMPILER_LAUNCHER=/usr/bin/ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=/usr/bin/ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=/usr/bin/ccache \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/clang++ \
    "-DCMAKE_CXX_FLAGS=-stdlib=libc++ -include cstdlib" \
    "-DCMAKE_EXE_LINKER_FLAGS=-stdlib=libc++" \
    -DCPPTRACE_DISABLE_CXX_20_MODULES=ON
