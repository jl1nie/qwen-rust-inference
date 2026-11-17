#!/usr/bin/env bash
# Build and run the CUDA BF16 GEMV benchmark (issue #46).
#
#   tools/bench_cuda_gemv/run.sh                # check + time, print only
#   tools/bench_cuda_gemv/run.sh out.json       # also save the log as JSON
#   CUDA_ARCH=sm_89 tools/bench_cuda_gemv/run.sh
#   BENCH_ARGS="--rows 1024 --cols 3584" tools/bench_cuda_gemv/run.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
cuda_root="${CUDA_PATH:-/usr/local/cuda}"
nvcc="$cuda_root/bin/nvcc"
if [[ ! -x "$nvcc" ]]; then
  printf '%s\n' "nvcc not found at $nvcc; run tools/setup-cuda-wsl.sh or set CUDA_PATH" >&2
  exit 1
fi
arch="${CUDA_ARCH:-native}"
# --fmad=false: only the explicit fmaf calls fuse, so the CPU emulation of the
# lane order is bit exact. -ffp-contract=off does the same for the host code.
flags=(-O3 -std=c++17 -arch="$arch" --fmad=false -lineinfo -Xcompiler -ffp-contract=off)
mkdir -p target/cuda
"$nvcc" "${flags[@]}" tools/bench_cuda_gemv/gemv.cu -o target/cuda/bench_cuda_gemv

toolkit=$("$nvcc" --version | grep -o 'release [^,]*, V[0-9.]*' || true)
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
kernel=$(uname -r)
args=(--driver "${driver:-unknown} (Windows host driver; WSL2 kernel $kernel)"
      --toolkit "${toolkit:-unknown}"
      --build "nvcc ${flags[*]}")
if [[ $# -ge 1 ]]; then args+=(--json "$1"); fi
# shellcheck disable=SC2086
target/cuda/bench_cuda_gemv "${args[@]}" ${BENCH_ARGS:-}
