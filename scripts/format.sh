#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"

cd "${project_dir}"

mapfile -t files < <(
    {
        git diff --name-only --diff-filter=ACMRTUXB
        git diff --name-only --diff-filter=ACMRTUXB --cached
        git ls-files --others --exclude-standard
    } | sort -u | grep -E '\.(c|cc|cpp|cxx|cu|h|hh|hpp|hxx)$' || true
)

if ((${#files[@]} == 0)); then
    exit 0
fi

clang-format -i -- "${files[@]}"
