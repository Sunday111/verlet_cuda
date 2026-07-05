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

"${script_dir}/configure.sh" "${build_dir}"

cmake --build "${build_dir}" --target verlet_cuda --parallel
cmake --build "${build_dir}" --target verlet_cuda_copy_files --parallel
