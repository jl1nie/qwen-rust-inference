#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# CUDA bridge (ADR 0033): the ROCm bridge source compiled by nvcc for NVIDIA
# GPUs. native/cuda maps the HIP/rocBLAS names onto CUDA/cuBLAS. On WSL2 install
# only the toolkit (tools/setup-cuda-wsl.sh); the driver comes from Windows.
# CUDA_ARCH defaults to the GPU in this machine (native); set e.g. sm_89 to
# build for another. --fmad=false: no automatic FMA contraction, as the HIP
# build's -ffp-contract=off (kernels write fmaf where they mean it).
cuda_root="${CUDA_PATH:-/usr/local/cuda}"
if [[ ! -x "$cuda_root/bin/nvcc" ]]; then
  printf '%s\n' "nvcc not found at $cuda_root/bin/nvcc; run tools/setup-cuda-wsl.sh or set CUDA_PATH" >&2
  exit 1
fi
arch="${CUDA_ARCH:-native}"
mkdir -p target/cuda
"$cuda_root/bin/nvcc" -std=c++17 -O2 --fmad=false -arch="$arch" -Xcompiler -ffp-contract=off \
  -x cu -DQR_CUDA -Inative/cuda -Inative/rocm \
  -shared -Xcompiler -fPIC native/rocm/bridge.cpp \
  -L"$cuda_root/lib64" -Xlinker -rpath -Xlinker "$cuda_root/lib64" -lcublas \
  -o target/cuda/libqwen_cuda.so
printf '%s\n' "Built target/cuda/libqwen_cuda.so (CUDA $arch, FP32 compute, BF16 weights)"
