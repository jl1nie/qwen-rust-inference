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
# Kernel sources (native/metal/kernels/, reading order, spec/metal.md) are
# embedded as C++ raw string literals; the .metal files stay the source of truth.
# Library 1 (levels 1-2, any Apple GPU): common + the op files in this order.
# Library 2 (level 3, Metal 4 tensor ops, optional at runtime): common + tensor.
embed() {
  printf 'R"QRMETAL('
  cat "$@"
  printf ')QRMETAL"\n'
}
k=native/metal/kernels
embed $k/common.metal $k/elementwise.metal $k/norm.metal $k/deltanet.metal \
  $k/attention_decode.metal $k/attention_prefill.metal $k/matmul.metal > target/metal/kernels.inc
embed $k/common.metal $k/tensor.metal > target/metal/kernels_tensor.inc
clang++ -std=c++17 -O2 -Wall -Wextra -Werror -fobjc-arc -ffp-contract=off -mmacosx-version-min=15.0 \
  -dynamiclib native/metal/bridge.mm -I target/metal \
  -framework Metal -framework Foundation \
  -o target/metal/libqwen_metal.dylib
printf '%s\n' 'Built target/metal/libqwen_metal.dylib (Metal, ABI v3, FP32 compute + BF16 weights, runtime-compiled kernels)'
