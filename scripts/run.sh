#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
build_dir="${1:-}"
if [[ -z "${build_dir}" ]]; then
    build_dir="${project_dir}/build"
fi
app="${build_dir}/bin/verlet_cuda"

"${script_dir}/build.sh" "${build_dir}"

if command -v prime-run >/dev/null 2>&1; then
    exec prime-run "${app}"
fi

export __NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD:-1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
export __VK_LAYER_NV_optimus="${__VK_LAYER_NV_optimus:-NVIDIA_only}"

exec "${app}"
