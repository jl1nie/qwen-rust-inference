#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Build on the Mac that will run it. Needs only the Command Line Tools
# (clang + macOS SDK); kernels are compiled at runtime by the Metal framework,
# so the separate Xcode `metal` shader compiler is NOT required.
if [[ "$(uname -s)" != Darwin ]]; then
  printf '%s\n' 'Metal bridge builds only on macOS' >&2
  exit 1
fi
mkdir -p target/metal
# Embed kernels.metal as a C++ raw string literal (source of truth stays .metal).
{
  printf 'R"QRMETAL('
  cat native/metal/kernels.metal
  printf ')QRMETAL"\n'
} > target/metal/kernels.inc
clang++ -std=c++17 -O2 -Wall -Wextra -Werror -fobjc-arc -ffp-contract=off -mmacosx-version-min=15.0 \
  -dynamiclib native/metal/bridge.mm -I target/metal \
  -framework Metal -framework Foundation \
  -o target/metal/libqwen_metal.dylib
printf '%s\n' 'Built target/metal/libqwen_metal.dylib (Metal, ABI v3, FP32 compute + BF16 weights, runtime-compiled kernels)'
