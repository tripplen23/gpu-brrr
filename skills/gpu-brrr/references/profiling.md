# Profiling & Regime Diagnosis

Use profiling to confirm the regime found in the diagnosis workflow. Do not guess.

## 0. Benchmark correctly

- Fix seeds and inputs where possible.
- Warm up eager and compiled paths independently.
- Call `torch.cuda.synchronize()` around wall-clock timing, or use CUDA events.
- Measure enough iterations to report median and p95 (or median and IQR).
- Keep compile/autotune time separate from steady-state time.
- Verify outputs before comparing speed.

## 1. The batch-size scaling test (cheap overhead clue)

Overhead is roughly constant with data size; compute and memory scale with it.

```python
import time, torch

def timed(fn, iters=50):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

# Measure at 1x and 2x batch. If time barely changes -> overhead-bound.
```

Interpretation:
- 2× data → ~+10% time  => overhead-bound hypothesis; confirm in a trace.
- 2× data → ~2× time    => compute- or memory-bound (measure FLOPS next).

This is not proof: larger batches can improve occupancy or arithmetic intensity.

## 2. PyTorch profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True) as prof:
    model(inputs)          # run a few steps
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")   # open in chrome://tracing or Perfetto
```

Reading the trace:
- **GPU kernels back-to-back, CPU running ahead** → launch overhead is mostly hidden;
  inspect kernel counters and synchronization next.
- **Gaps between GPU kernels while the CPU row is busy** → CPU/framework can't keep the
  GPU fed → **overhead-bound**.
- Many tiny pointwise/elementwise kernels are a **fusion candidate**, but use counters
  to confirm bandwidth pressure.
- One or few large matmul/`gemm`/`conv` kernels are likely compute-heavy, but inefficient
  shapes or low occupancy can still leave them below the compute roof.

## 3. nvidia-smi quick eyeball

```
nvidia-smi           # or: nvidia-smi dmon
```
- **GPU-Util** ≈ fraction of wall time a GPU kernel was running. Low GPU-Util with a
  busy CPU strongly suggests overhead-bound.
- **Memory (DRAM) usage** shown here is capacity (the thing that causes CUDA OOM), NOT
  bandwidth. High DRAM usage says nothing about whether you're bandwidth-bound.

## 4. Kernel-level tools (deeper)

- `torch.compile(model, mode="max-autotune")` then profile again to compare.
- Nsight Systems (`nsys`) for full timeline; Nsight Compute (`ncu`) for per-kernel
  achieved bandwidth and FLOPS vs peak (the ground truth for compute-vs-memory bound).
- `ncu` reports memory and compute throughput. Use the dominant saturated resource plus
  stall reasons to classify a kernel; do not infer the whole model from one kernel.

## Decision summary

1. Batch-scaling flat + timeline gaps? → overhead-bound.
2. High DRAM throughput + low compute throughput? → bandwidth-bound.
3. High compute throughput using the matching dtype peak? → compute-bound.
4. Neither resource is busy? → investigate shapes, occupancy, dependencies, graph
   breaks, synchronization, input, or communication before assigning a regime.
