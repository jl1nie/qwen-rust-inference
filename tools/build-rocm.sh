#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Build on the target WSL machine after installing a matched AMD stack.
# Explicit GPU target keeps compilation reproducible; RX 7900 XTX is gfx1100.
rocm_root="${ROCM_PATH:-/opt/rocm}"
if [[ ! -x "$rocm_root/bin/hipcc" ]]; then
  printf '%s\n' "hipcc not found at $rocm_root/bin/hipcc; install the matched WSL ROCm development stack or set ROCM_PATH" >&2
  exit 1
fi
mkdir -p target/rocm
"$rocm_root/bin/hipcc" -std=c++17 -O2 -ffp-contract=off --offload-arch=gfx1100 \
  -shared -fPIC native/rocm/bridge.cpp -I"$rocm_root/include" \
  -L"$rocm_root/lib" -Wl,-rpath,"$rocm_root/lib" -lrocblas \
  -o target/rocm/libqwen_rocm.so
printf '%s\n' 'Built target/rocm/libqwen_rocm.so (HIP, gfx1100, FP32 compute, BF16 weights)'
