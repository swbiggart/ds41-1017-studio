#!/bin/bash
# Install the pinned Zig 0.17.0 toolchain for Apple Silicon Macs.
set -euo pipefail
mode=${1:---zig-only}
if [[ $# -gt 1 || $mode != --zig-only ]]; then
  echo "Usage: $0 [--zig-only]" >&2
  exit 2
fi
root=$(cd "$(dirname "$0")/.." && pwd)
local_dir="$root/.local"
cd "$root"
zig_dir="$local_dir/zig-pypi"
zig="$zig_dir/ziglang/zig"
if [[ $(uname -sm) != "Darwin arm64" ]]; then
  echo "This recipe requires an Apple Silicon Mac." >&2
  exit 1
fi
mkdir -p "$local_dir"
if [[ ! -x "$zig" ]]; then
  archive="$local_dir/ziglang-0.17.0.whl"
  curl --fail --location --retry 3 --output "$archive" \
    'https://files.pythonhosted.org/packages/37/fa/54dd03b21fa790a85e1de39eeda9703386fe519490598abdb38894a1c3b3/ziglang-0.17.0-py3-none-macosx_13_0_arm64.whl'
  echo "87274af689c4128051b26c80f6f676757e0a81c2fdc137fd8b5679e928e4bc4f  $archive" | shasum -a 256 -c -
  stage=$(mktemp -d "$local_dir/zig-stage.XXXXXX")
  trap 'rm -rf "$stage"' EXIT
  unzip -q "$archive" -d "$stage"
  mkdir -p "$zig_dir"
  mv "$stage/ziglang" "$zig_dir/"
fi
[[ $("$zig" version) == 0.17.0 ]]
