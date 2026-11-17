# bench_cuda_gemv

A standalone CUDA benchmark of the single-position BF16 projection, built for
issue #46 and chapter 8 of book vol. 2. It is not linked into the Rust engine.

```text
y[rows] = W[rows, cols] · x[cols]     W: row-major BF16, x and y: FP32
one warp per output row; lanes accumulate in FP32 with fmaf,
then __shfl_down_sync at distances 16, 8, 4, 2, 1; lane 0 writes y[row]
```

```bash
sudo tools/setup-cuda-wsl.sh                  # once: CUDA Toolkit for WSL2 (no driver)
tools/bench_cuda_gemv/run.sh                  # build, check, time 3584x1024
tools/bench_cuda_gemv/run.sh out.json         # also save the log
BENCH_ARGS="--rows 1024 --cols 3584" tools/bench_cuda_gemv/run.sh
BENCH_ARGS=--check-only tools/bench_cuda_gemv/run.sh
```

`CUDA_PATH` (default `/usr/local/cuda`) and `CUDA_ARCH` (default `native`)
override the toolkit and the target. The binary is `target/cuda/bench_cuda_gemv`.
It exits with a non-zero status if any check fails; timing runs only after all
checks pass.

## Kernels

Each kernel makes one change from the one before it:

| name | weight loads per lane | order of the elements a lane owns |
|---|---|---|
| `scalar` | one BF16 (2 bytes) per step | `j = lane, lane+32, ...` |
| `vec8` | one `uint4` (8 BF16, 16 bytes) per step | pieces `p = lane, lane+32, ...`, elements `8p..8p+7` |
| `pf2` / `pf4` | 2 / 4 `uint4` loads issued before their FMAs | same as `vec8` (bitwise equal) |

`vec8` and `pf*` need 16-byte aligned rows: `cols % 8 == 0` and an aligned
base pointer. Each row checks this, and an unaligned row takes the `scalar` loop.

## Checks

Inputs are deterministic (SplitMix64; BF16 by round-to-nearest-even). The CPU
and the GPU use the same BF16 bit patterns and FP32 inputs. Cases:

- all ones, 3584x1024: every row must be exactly 1024;
- signed non-uniform values in [-1, 1), 3584x1024 and 1024x3584;
- small widths that are not multiples of 32: 45, 100 (`% 8 = 4`), 1000
  (`% 256 != 0`, tail of `pf*`), 8, 1; rows that are not multiples of 8;
- the weight base offset by 2 and by 8 bytes (16-byte loads not allowed);
- 64 sentinel slots after `y[rows-1]` must stay untouched.

Each kernel's output must pass two checks:

1. It equals, bit for bit, a CPU emulation of that kernel's lane order and
   shuffle tree (`std::fmaf`, host code built with `-ffp-contract=off`,
   device code with `--fmad=false`).
2. **Tolerance.** `|gpu - ref| <= atol + rtol * Σ_j |w_j x_j|` with
   `atol = 1e-6`, `rtol = 1e-5`, where `ref` is accumulated in double.
   A lane adds at most ⌈cols/32⌉ products and the tree adds 5 more terms, so
   the FP32 rounding error is bounded by about `(cols/32 + 5) · 2^-24 · Σ|w x|`.
   That is ≤ 7e-6 · Σ|w x| for cols ≤ 3584. `atol` covers rows whose sum
   cancels to near zero.

The log prints the largest absolute difference, the largest error as a
fraction of the tolerance, and the row where it occurs (GPU value and
reference).

## Timing

- **Kernel time.** 200 launches are captured once into a CUDA Graph, and one
  `cudaEvent` pair is placed around each replay. 15 samples are taken after
  warmup, and the log reports the median and the range per launch. On WSL2 an
  ordinary launch costs about 9 µs on the device timeline (empty kernel), but
  a graph node costs about 0.8 µs. Without the graph, the hot case would
  mostly measure launch cost.
  - **hot**: the same 7 MB of weights every launch (stays in the 24 MB L2).
  - **cold**: copies of the weights cycled until the working set is at
    least max(8 × L2, 256 MiB), i.e. 37 copies and 272 MB for 3584x1024.
- **With issue cost** (reported separately): the same loops with ordinary
  stream launches, the host time to enqueue them, and one launch followed by
  `cudaStreamSynchronize`.
- **Raw read**: a read-only grid-stride `uint4` kernel over the cold set, the
  attainable DRAM read rate to compare the cold GEMV against.
- **Weight bandwidth** = `2 × rows × cols / seconds per launch`
  (7,340,032 bytes for 3584x1024). This counts weight bytes only, not bytes
  measured by a profiler. Nsight Compute under WSL2 needs GPU counter access,
  which is enabled on the Windows side (NVIDIA Control Panel → Developer →
  Manage GPU Performance Counters). Without it `ncu` reports `ERR_NVGPUCTRPERM`.

Results and their discussion are in [spec/cuda.md](../../spec/cuda.md).
