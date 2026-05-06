# Performance Notes

This file summarizes the profiling and optimization work done on `infergpt` so future agents can continue performance work without reconstructing the thread.

## What We Are Profiling

- Executable: [infergpt/build/src/infergpt](/home/toby/dev/learngpt/infergpt/build/src/infergpt)
- Workload: greedy GPT-2 small / 124M inference on the repository's fixed tokenized prompt in [input.txt](/home/toby/dev/learngpt/input.txt), generating the 40-token continuation in [output.txt](/home/toby/dev/learngpt/output.txt)
- Model shape: GPT-2 small style, `n_embd=768`, `n_head=12`, `n_layer=12`, `n_vocab=50257`
- Decode regime: single sequence, prompt length `10`, then repeated full forward passes for lengths `10..49`
- Important implication: this is not a large batched GEMM workload; it is many relatively small affine/matmul-like operations during autoregressive decoding

## Measurement Workflow

### End-to-end timing

Primary command used:

```bash
/usr/bin/time -f 'elapsed=%e user=%U sys=%S maxrss=%M' infergpt/build/src/infergpt >/tmp/infergpt.stdout 2>/tmp/infergpt.time
```

### Affine microbenchmark

- Benchmark source: [infergpt/bench/affine_bench.cpp](/home/toby/dev/learngpt/infergpt/bench/affine_bench.cpp)
- Helper script: [scripts/bench_affine.sh](/home/toby/dev/learngpt/scripts/bench_affine.sh)
- Cases covered:
  - `attn_qkv`: `N x 768 -> N x 2304`
  - `attn_proj`: `N x 768 -> N x 768`
  - `mlp_fc`: `N x 768 -> N x 3072`
  - `mlp_proj`: `N x 3072 -> N x 768`
- Default row counts: `10`, `25`, `49`

### Vector-matrix scaling benchmark

- Benchmark source: [infergpt/bench/vector_scaling_bench.cpp](/home/toby/dev/learngpt/infergpt/bench/vector_scaling_bench.cpp)
- Purpose: compare serial vs parallel `RowVector x Matrix^T` performance across decode-relevant shapes, thread counts, and output block sizes
- Cases covered:
  - `decode_logits`: `1 x 768 -> 1 x 50257`
  - `decode_attn_10`, `decode_attn_25`, `decode_attn_49`: `1 x 64 -> 1 x N`
  - `mlp_fc`: `1 x 768 -> 1 x 3072`
  - `mlp_proj`: `1 x 3072 -> 1 x 768`

### Perf profiling

- Helper script: [scripts/profile_affine.sh](/home/toby/dev/learngpt/scripts/profile_affine.sh)
- Main tools used:
  - `perf record -g`
  - `perf report --stdio`
  - `perf annotate --stdio`
  - `gprofng` for one full-run sample set

### Compiler/codegen inspection

- Symbol lookup: `nm -C`
- Disassembly: `objdump -d -C -Mintel`
- Vectorization diagnostics: clang `-Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize`

## Original and Current Timings

Measured milestones from this thread:

| Stage | Approx time |
| --- | ---: |
| Original baseline before current optimization pass | `~168s` |
| After shifting affine work to `X * Y^T` world and removing the final accidental generic multiply path | `~49.0s` |
| After computing only the needed last logits row in `Model::Forward` | `34.76s` |
| After `-ffast-math` only | `6.08s` |
| After `-ffast-math -mavx2 -mfma` | `4.43s` |
| Reconfirmed current best `-ffast-math -mavx2 -mfma` build | `4.49s` |
| `-ffast-math -mavx512f -mfma` | `4.61s` |
| First parallel `MatMul_XYT` baseline | `2.21s` |
| Parallel 2D tile `8 x 64` | `2.02s` |
| Parallel 2D tile `12 x 64` | `~1.84s` to `1.97s` |
| Parallel 2D tile `16 x 32` | `~1.70s` to `1.85s` |
| Parallel 2D tile `16 x 64` | `1.96s` |
| Reconfirmed reverted `16 x 32` auto-vectorized build | `1.72s` warmed, `1.87s` cold |

Overall improvement from the original baseline to the current kept build is about `168s -> ~1.7s`, roughly `98x` faster.

## Major Optimization Steps

### 1. Affine kernel/data-layout change: compute `X * Y^T`

Key idea:

- `Affine` now stores transposed weights as `weights_T`
- The hot path computes `MatMul_XYT(x, weights_T)` instead of a generic `X * Y`
- This matches row-major storage better and improved locality for the dot-product-style inner loop

Relevant code:

- [infergpt/src/layers.h](/home/toby/dev/learngpt/infergpt/src/layers.h)
- [infergpt/src/algebra.h](/home/toby/dev/learngpt/infergpt/src/algebra.h#L58)

Measured result:

- Brought the end-to-end run down from roughly `168s` to the `49s` range after the full switch-over and cleanup of the last generic multiply use

Related correctness fixes discovered along the way:

- `Matrix` assignment/copy semantics had left cached `ptr_` stale after some operations
- `MatMul_XYT` had an output-shape bug that previously caused a heap-buffer overflow in tests

These were required to make the optimized path correct and testable.

### 2. Stop computing unused rows in the final projection

Key idea:

- During greedy generation, only the final token's logits are used
- [infergpt/src/model.h](/home/toby/dev/learngpt/infergpt/src/model.h#L41) was changed so `Model::Forward` computes only the last normalized row times `wte_^T`, not the full sequence's logits matrix

Measured result:

- `49.0s -> 34.76s`

Behavior check:

- Generated output remained identical to [output.txt](/home/toby/dev/learngpt/output.txt) on the fixed workload

### 3. Release compile flags to unlock and widen auto-vectorization

Current best-known Release flags in [infergpt/CMakeLists.txt](/home/toby/dev/learngpt/infergpt/CMakeLists.txt#L8):

```cmake
add_compile_options(
  $<$<CONFIG:Release>:-ffast-math>
  $<$<CONFIG:Release>:-mavx2>
  $<$<CONFIG:Release>:-mfma>
)
```

Measured result:

- `34.76s -> 6.08s` from `-ffast-math`
- `6.08s -> 4.43s` from adding `-mavx2 -mfma`

This was the largest single incremental speedup after the code shape had been cleaned up enough for the compiler to see the reduction pattern.

## What We Learned About Auto-Vectorization

### Why clang originally did not vectorize the hot matmul reduction

The key loop in [infergpt/src/algebra.h](/home/toby/dev/learngpt/infergpt/src/algebra.h#L83) is:

```cpp
for (int k = 0; k < lcols; ++k)
  sum += pl[k] * pr[k];
```

Clang's vectorization remarks explicitly reported:

- `loop not vectorized: cannot prove it is safe to reorder floating-point operations`

Interpretation:

- This was not an unusual hidden `fp:strict` mode
- It was the normal default precise IEEE-style floating-point behavior preventing reassociation of the reduction
- Because vectorizing the loop changes the order of additions, clang refused without fast-math-style permission

### What changed with `-ffast-math`

- Under `-ffast-math`, clang changed the same loop from missed to vectorized
- With generic baseline ISA, the loop became packed XMM code
- With `-mavx2 -mfma`, clang reported width `8`, interleave `4`
- With `-mavx512f -mfma`, clang reported width `16`, interleave `4`

### Disassembly findings

Before `-ffast-math`:

- The hot loop in `MatMul_XYT` used scalar `mulss` / `addss` on XMM registers, just unrolled a little
- It was not true packed SIMD math

After `-ffast-math -mavx2 -mfma`:

- The hot loop used YMM packed vector code with `vmovups`, `vfmadd231ps`, `vaddps`, and scalar cleanup

After `-ffast-math -mavx512f -mfma`:

- The hot loop used ZMM packed vector code with `vmovups zmm*`, `vfmadd231ps zmm*`, and `vaddps zmm*`

### Important fast-math caveat

- Whole-program `-ffast-math` affects semantics globally
- A test using `-std::numeric_limits<float>::infinity()` triggered a warning under fast-math; this was changed to `lowest()` in [infergpt/tests/layers_test.cpp](/home/toby/dev/learngpt/infergpt/tests/layers_test.cpp#L117)
- For this inference-only program, fast-math was judged acceptable because there is no exact canonical reference and small rounding-order differences are acceptable

## Compiler Flag Experiments

### Neutral or bad before auto-vectorization was enabled

These were tried before the final fast-math breakthrough:

- `-march=native`: much worse, about `68.04s`
- `-mavx2 -mfma`: much worse, about `67.91s`
- `-mavx2 -mfma -mavx512f -mavx512vl`: much worse, about `67.85s`
- `-msse3`: neutral, about `34.72s`
- `-mavx`: neutral, about `34.74s`
- `-mavx2`: neutral, about `34.73s`

Interpretation:

- Wider ISA flags alone did not help while the compiler still refused to vectorize the reduction due to FP semantics
- Once `-ffast-math` was enabled, ISA width mattered a lot

### AVX2 vs AVX-512 after fast-math

- `-ffast-math -mavx2 -mfma`: `4.43s`
- `-ffast-math -mavx512f -mfma`: `4.61s`

Interpretation:

- AVX-512 widened the loop but was still slightly slower on this CPU
- Most likely explanation is the usual tradeoff: wider vectors, but worse effective frequency / other overheads
- Best tested configuration remains AVX2 + FMA

### FMA4 curiosity check

Tried for curiosity only:

- `-ffast-math -mavx2 -mfma4`
- `-ffast-math -mavx512f -mfma4`

Findings:

- Both builds compiled but crashed at runtime with `SIGILL` on this machine
- A separate object-level comparison showed clang does generate real FMA4 instructions, but the `MatMul_XYT` kernel was not tighter; the FMA4 version was slightly larger than the FMA3 version
- Conclusion: FMA4 is not useful here and not supported by the current CPU

## Python Reference Comparison

Reference used:

- [picoGPT/gpt2.py](/home/toby/dev/learngpt/picoGPT/gpt2.py), specifically `decode_output_file()` to regenerate the exact token continuation stored in [output.txt](/home/toby/dev/learngpt/output.txt)

Measured result:

- Python reference runtime: `14.53s`
- Current C++ runtime: `4.49s`

So the current C++ implementation is about `3.2x` faster than the Python reference on the same workload.

Threading observation:

- The Python process peaked at `33` threads during execution
- NumPy in the venv is linked against OpenBLAS (`scipy-openblas` / OpenBLAS 0.3.31)
- No `OPENBLAS_NUM_THREADS` or `OMP_NUM_THREADS` limit was set
- Interpretation: the Python reference is likely using a 32-thread OpenBLAS worker pool on this 32-core machine

This remained notable even before adding C++ multithreading: the optimized single-threaded C++ implementation was already faster than the Python reference while Python was using a 32-thread OpenBLAS pool.

## Parallelism Findings

### Thread-pool baseline

Relevant code:

- [infergpt/src/pfor.h](/home/toby/dev/learngpt/infergpt/src/pfor.h)
- [infergpt/src/algebra.h](/home/toby/dev/learngpt/infergpt/src/algebra.h)
- [infergpt/bench/thread_scaling_bench.cpp](/home/toby/dev/learngpt/infergpt/bench/thread_scaling_bench.cpp)

The first parallel version used a lightweight custom `ThreadPool` and parallelized `MatMul_XYT` over flattened `(lrow, rrow-block)` work items.

Measured result:

- End-to-end runtime dropped from about `4.49s` to `2.21s`
- Generated output remained identical to [output.txt](/home/toby/dev/learngpt/output.txt)

Important interpretation:

- This was a meaningful latency win, but still far from linear scaling on a 32-core machine

### Why the initial 32-core speedup was limited

`perf stat` on the first parallel build showed:

- elapsed time about `2.2157s`
- task-clock about `30.3s`
- average CPU utilization about `13.695 CPUs`
- about `140.6B` cycles and `46.8B` instructions
- about `85,651` context switches

Interpretation:

- The machine has 32 cores, but the workload only sustained about 13.7 CPUs on average
- The issue was not just scheduler overhead; decode-time affine shapes are small enough that memory bandwidth, cache behavior, and limited work per task all matter

### Scaling benchmark and what it showed

To make scaling experiments easier, a dedicated benchmark was added:

- [infergpt/bench/thread_scaling_bench.cpp](/home/toby/dev/learngpt/infergpt/bench/thread_scaling_bench.cpp)

This benchmark sweeps thread counts and tile sizes for representative affine shapes.

Representative findings from this thread:

- `mlp_fc`, `rows=49`, untiled parallel baseline, `rrows_per_block=64`:
  - `1 thread`: `85.35 GFLOP/s`
  - `32 threads`: `596.92 GFLOP/s`
  - roughly `7x` scaling, not `32x`
- `attn_proj`, `rows=10`, untiled baseline:
  - scaled well up to mid thread counts, then flattened or regressed by 32 threads
- Sweeps over `rrows_per_block = 1, 8, 16, 32, 64, 128` showed that block size affected results at the margin, but did not remove the scaling ceiling

Conclusion:

- The dominant limit was not a single bad block-size choice; it was the combination of small decode-time problem sizes and cache/bandwidth pressure

### Simple `K`-blocking was not a win

A straightforward attempt was made to add `K`-blocking inside each existing parallel task.

Variants tried:

- `kcols_per_block = 256`
- `kcols_per_block = 768`

Measured result:

- `256` made the full run worse: about `2.25s`
- `768` recovered most of the loss, but still did not beat the simpler baseline: about `2.21s`

Interpretation:

- Naive `K`-blocking added extra accumulation and loop overhead without improving locality enough to matter
- If deeper tiling is pursued in future, it should probably be a small accumulator or microkernel-style approach rather than just splitting `K`

### Simple 2D output tiling did help

The next experiment changed the parallel work shape from one `lrow` at a time to a small 2D output tile:

- `lrows_per_block x rrows_per_block`
- the inner `K` reduction for each output cell was otherwise unchanged

Measured full-run results from the tuning pass:

- `4 x 64`: `2.07s`
- `6 x 64`: `2.03s`
- `8 x 64`: `2.02s`
- `12 x 32`: `1.87s`
- `12 x 64`: observed in the `1.84s` to `1.97s` range
- `16 x 32`: observed in the `1.70s` to `1.85s` range
- `16 x 64`: `1.96s`
- `12 x 128`: `2.09s`

An interleaved A/B rerun gave a clearer result than the original one-off timings:

- `12 x 64`: 6 runs, average `1.858s`, range `1.83s` to `1.97s`
- `16 x 32`: 6 runs, average `1.710s`, range `1.70s` to `1.72s`

All repeated runs preserved exact output.

The best tuned kernel configuration currently in [infergpt/src/algebra.h](/home/toby/dev/learngpt/infergpt/src/algebra.h) is:

- `lrows_per_block = 16`
- `rrows_per_block = 32`

Why this likely helped:

- Reusing the same `rhs` block across multiple `lhs` rows improved locality enough to matter in the real decode workload
- The best end-to-end setting was not the same one that looked best on every isolated microbenchmark, which suggests the smaller decode-time shapes matter disproportionately for real latency

### Practical current state

Current best measured end-to-end runtime is the reverted parallel tiled build with a `16 x 32` tile, observed in the `1.70s` to `1.85s` range, with exact output preserved on the repository workload.

A fresh rerun after removing the later experiments gave:

- cold run: `1.87s`
- warmed runs: `1.72s`, `1.72s`, `1.72s`
- all runs matched [output.txt](/home/toby/dev/learngpt/output.txt)

### Later experiments were rolled back

After the `16 x 32` auto-vectorized path was already running at about `1.71s`, additional complexity was tried on top:

- hand-written AVX2/FMA microkernels
- packed `rhs` / outer-product layout experiments

Those experiments did not produce a meaningful enough end-to-end gain to justify their maintenance cost, so they were rolled back.

The repository's kept state is therefore the simpler auto-vectorized `MatMul_XYT` implementation in [infergpt/src/algebra.h](/home/toby/dev/learngpt/infergpt/src/algebra.h), using the `16 x 32` parallel tile and no custom microkernel code.

## Current State and Likely Next Work

Current best-known build state:

- [infergpt/CMakeLists.txt](/home/toby/dev/learngpt/infergpt/CMakeLists.txt#L8) uses `-ffast-math -mavx2 -mfma` for Release builds
- [infergpt/tests/layers_test.cpp](/home/toby/dev/learngpt/infergpt/tests/layers_test.cpp#L117) uses `lowest()` instead of `-infinity()` for fast-math friendliness
- [infergpt/src/algebra.h](/home/toby/dev/learngpt/infergpt/src/algebra.h) currently uses the simpler parallel `16 x 32` output tile inside `MatMul_XYT`, relying on compiler auto-vectorization rather than a hand-written microkernel

## Vector-Matrix (`RowVector x Matrix^T`) Follow-Up

This pass looked at the vector-matrix overload of [infergpt/src/algebra.h](/home/toby/dev/learngpt/infergpt/src/algebra.h), which is now a major decode-time hotspot after splitting prefill and decode.

### Obvious issue found first

The original vector-matrix path always called `ParallelFor` and always used a fixed output block of `16 * 32 = 512` rows.

That meant:

- tiny decode-attention shapes like `1 x 64 -> 1 x 49` still paid thread-pool overhead
- medium and large decode projections could not choose a better block size by shape

### Focused benchmark findings

The dedicated benchmark in [infergpt/bench/vector_scaling_bench.cpp](/home/toby/dev/learngpt/infergpt/bench/vector_scaling_bench.cpp) was used to sweep serial vs parallel, threads, and output block size.

Representative results:

- `decode_attn_49` (`1 x 64 -> 1 x 49`):
  - serial: about `46.5 GFLOP/s`
  - all parallel variants were much worse, typically about `2` to `15 GFLOP/s`
  - conclusion: these tiny decode-attention products should stay serial
- `mlp_fc` (`1 x 768 -> 1 x 3072`) at `32` threads:
  - serial: about `43 GFLOP/s`
  - best observed parallel block: `64`, about `215 GFLOP/s`
  - `512` was materially worse, about `157 GFLOP/s`
- `mlp_proj` (`1 x 3072 -> 1 x 768`) at `32` threads:
  - serial: about `54 GFLOP/s`
  - best observed parallel block: `16`, about `218 GFLOP/s`
  - larger blocks regressed steadily
- `decode_logits` (`1 x 768 -> 1 x 50257`) at `32` threads:
  - serial: about `24 GFLOP/s`
  - parallel variants clustered around `27` to `28 GFLOP/s`
  - best observed blocks were in the rough `128` to `1024` range, with only modest differences between them

### Kept kernel shape

The current kept vector-matrix path in [infergpt/src/algebra.h](/home/toby/dev/learngpt/infergpt/src/algebra.h) remains the simpler baseline:

- always uses `ParallelFor`
- uses a fixed output block size of `16 * 32 = 512`

Interpretation:

- the focused benchmark did show that some isolated vector-matrix cases preferred a serial path or smaller block sizes
- but the attempted heuristic dispatch based on those microbenchmarks was worse on the real end-to-end workload and was reverted

### Codegen check

Disassembly of the row-vector `MatMul_XYT` path in the Release AVX2/FMA build showed the expected packed SIMD reduction:

- `vmovups ymm*`
- `vfmadd231ps ymm*`
- `vaddps`
- scalar cleanup with `vfmadd231ss`

So the vector-matrix inner loop is still auto-vectorized and using FMA in the same broad style as the matrix-matrix kernel.

### End-to-end effect so far

The heuristic vector-matrix dispatch based on a serial cutoff and shape-selected block sizes did not improve the real workload and was worse than the simpler baseline, so it was reverted.

Interpretation:

- the isolated vector-matrix kernels did show interesting microbenchmark behavior
- but those results did not transfer cleanly to the end-to-end run
- the row-vector `MatMul_XYT` path is still worth attention because profiling shows it remains one of the dominant compute hotspots, but future changes should be validated primarily against the full decode workload rather than microbenchmarks alone

What is probably left on the table for single-thread performance:

- better blocking / tiling inside the affine kernel
- computing multiple output columns per inner pass to improve reuse of the left row in cache/registers
- reducing temporary allocation and copy traffic between layer boundaries

What is most promising overall from here:

- keep the current `16 x 32` auto-vectorized path unless a future change shows a clearly larger end-to-end win for its added complexity
- fresh profiling on the current tiled build to verify whether `MatMul_XYT` still dominates or whether another part of the decode loop is now next

Caution for future parallel work:

- do not schedule one tiny task per output column
- the decode workload has small row counts (`10..49`), so the granularity needs to be coarse enough that scheduling overhead does not dominate
- do not assume a microbenchmark winner is automatically the end-to-end winner; in this workload, the small decode-time shapes can shift the best tile choice
