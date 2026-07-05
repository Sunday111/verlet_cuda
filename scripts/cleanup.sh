#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"

cd "${project_dir}"

git submodule deinit --force --all
git clean -ffdX
git submodule update --init --recursive --force
