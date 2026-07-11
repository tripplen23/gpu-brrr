# FLOP Counting & Achieved-vs-Peak FLOPS

Achieved FLOPS as a percentage of the matching hardware peak is useful evidence, not a
standalone diagnosis. Pair it with kernel-level compute and memory throughput.

## 1. Count FLOPs in a PyTorch model

Modern PyTorch has a built-in FLOP counter via TorchDispatch:

```python
from torch.utils.flop_counter import FlopCounterMode

flop_counter = FlopCounterMode(model, display=True)
with flop_counter:
    model(inputs)            # forward
    # for training FLOPs, also run loss.backward()
total_flops = flop_counter.get_total_flops()
```

For a manual estimate: a matmul of `[M, K] @ [K, N]` costs `2 * M * K * N` FLOPs
(multiply + add). Matmuls dominate total FLOPs; pointwise/norm ops are negligible in
FLOP count even when they dominate *time*.

Rule of thumb for a dense transformer training step:
`FLOPs ≈ 6 * N_params * N_tokens` (fwd = 2×, bwd = 4×).

## 2. Measure achieved FLOPS

```python
achieved_flops = total_flops / seconds_per_step
```

Compare to the exact applicable peak from the current vendor spec:

- Match dtype: FP32, TF32, BF16, FP16, and FP8 have different roofs.
- Match dense vs structured-sparse claims.
- Match Tensor Core eligibility and count FMA consistently.
- Record whether the quoted number is boost/peak or a sustained measured roof.

A workload full of pointwise operations can show low achieved FLOPS while the GPU is
busy. Confirm high DRAM throughput before calling it memory-bound.

## 3. Measuring the memory-vs-compute crossover yourself

Reproduce the article's experiment: increase compute per byte ("compute intensity")
without increasing memory accesses, and watch runtime.

```python
def f(x, repeat):
    for _ in range(repeat):
        x = x * 2
    return x
```

For tensor of size `N` and `repeat` iterations:
- memory accesses ≈ `2 * N` (one read, one write of the tensor) — independent of repeat.
- FLOP ≈ `N * repeat`.
- achieved bandwidth = `bytes_per_elem * 2 * N * iters_per_sec`
- achieved FLOPS = `N * repeat * iters_per_sec`

Benchmark under a fusing compiler (`torch.compile`) across `repeat = 1,2,4,...,256` and
plot runtime / FLOPS / bandwidth vs compute intensity on log-log axes. You will see:
- Runtime nearly flat until ~64 multiplies (memory-bound; compute idle).
- Achieved FLOPS rises linearly with intensity until it plateaus near peak (compute-bound).
- Achieved bandwidth starts near peak and falls as intensity rises.

The crossover (~repeat 32–64 on an A100) marks the boundary between memory-bound and
compute-bound for that op.

## 4. Interpreting the number

- High achieved FLOPS and high compute-pipeline utilization → compute-bound with high
  confidence.
- Low achieved FLOPS + high DRAM throughput → memory-bound.
- Low achieved FLOPS + low DRAM and compute throughput → unclassified; inspect
  occupancy, shapes, stalls, synchronization, graph breaks, communication, and input.
