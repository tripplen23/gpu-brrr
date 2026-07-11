---
name: gpu-brrr
description: "Diagnoses and optimizes GPU performance for deep learning from first principles. Use for slow training or inference, low GPU utilization, CUDA OOM, PyTorch or Nsight profiling, torch.compile, kernel fusion, activation checkpointing, mixed precision, Triton kernels, launch/Python overhead, memory bandwidth, FLOPS utilization, roofline analysis, or classifying workloads as compute-, bandwidth-, or overhead-bound. Do not use for CPU-only optimization or model-quality tuning unless GPU performance is the bottleneck."
license: MIT
metadata:
  author: binh-nguyen
  version: '0.0.1'
  source: https://horace.io/brrr_intro.html
---

# GPU Brrr — Deep Learning Performance Optimization (First Principles)

Diagnose *why* a deep learning workload is slow before optimizing it. This skill gives
an AI agent a measurement-first workflow to identify the performance regime and apply
the matching optimization.

## When to Use This Skill

Use this skill when the user wants to:

- Speed up slow model training or inference.
- Understand why GPU utilization is low or the GPU "isn't going brrrr".
- Decide between fixes like `torch.compile`, kernel fusion, larger batch sizes,
  mixed precision, activation checkpointing, or rewriting hot paths.
- Profile a model and interpret the results.
- Reason about FLOPS, memory bandwidth, and framework/Python overhead.

Do NOT jump to a fix (fusion, tracing, bigger GPU) before diagnosing the regime.

## Core Mental Model

Use three categories to reason about accelerator time:

| Regime | What time is spent on | Analogy (factory) |
|--------|----------------------|-------------------|
| **Compute** | Actual FLOPs on the GPU (matmuls dominate) | The factory doing work |
| **Memory bandwidth** | Moving tensors between DRAM (global memory) and compute units (SRAM) | Shipping materials to/from the warehouse |
| **Overhead** | Everything else: Python interpreter, framework dispatch, kernel launch | Deciding what to build & issuing instructions |

Real models are usually mixed workloads. Diagnose the whole step and the dominant
kernels separately; do not force every trace into one label.

Key first-principles facts:

- For a fixed algorithm and numerical formulation, arithmetic is often less reducible
  than memory traffic and launch/framework overhead. Optimize useful throughput, not
  utilization counters alone.
- Modern accelerators have specialized matmul hardware. An A100 peaks near 312 TFLOPS
  for dense BF16 Tensor Core matmuls, versus about 19.5 TFLOPS on FP32 CUDA cores.
  Non-matmul operations (layernorm, activations, pointwise) contributed only ~0.2% of
  FLOPs in one BERT analysis yet often dominate *time* because they are memory-bound.
- Arithmetic throughput grows faster than memory bandwidth over hardware generations, so
  workloads drift toward being **memory-bound** over time.
- GPUs execute **asynchronously**, allowing PyTorch to queue kernels. Framework overhead
  may be hidden while the CPU stays ahead, but becomes exposed when it cannot keep the
  GPU fed.

## Diagnosis Workflow

### Step 0 — Define and measure the target

- Record hardware, software versions, dtype, shapes, batch size, mode
  (training/prefill/decode), and the exact metric: latency, throughput, step time,
  tokens/s, or cost.
- Separate cold-start compilation/autotuning from steady-state execution.
- Warm up, synchronize correctly, run multiple samples, and report median plus a tail
  percentile or dispersion. Preserve one untouched baseline.
- Check non-kernel blockers first: input pipeline, host-to-device copies, distributed
  communication, synchronization, allocator pressure, and data-dependent graph breaks.

### Step 1 — Identify the regime with the batch-size / size test

Fixed per-launch overhead usually scales less with data size than compute and memory
work. Use this as a cheap initial test:

- Increase batch size (or tensor size). Measure runtime.
- If runtime barely increases, treat **overhead-bound as a hypothesis**, not a verdict.
  Larger shapes can also improve occupancy or arithmetic intensity.
- If runtime scales roughly proportionally, compute or memory work may dominate; continue
  to Step 2.

### Step 2 — Measure achieved FLOPS vs peak FLOPS

- Compute achieved FLOPS = work / steady-state time. Compare against the matching
  dense/sparse, dtype, and Tensor Core peak; never mix FP32 work with BF16 peak.
- High achieved FLOPS plus high compute-pipeline utilization supports
  **compute-bound**.
- Low achieved FLOPS is inconclusive by itself. Confirm **memory-bandwidth-bound** with
  high DRAM throughput and low compute-pipeline utilization. Poor shapes, low occupancy,
  dependencies, synchronization, communication, or graph breaks can also produce low
  FLOPS.

Use `references/flop-counting.md` for how to count FLOPs and measure achieved FLOPS in
PyTorch (torch.utils.flop_counter / TorchDispatch), and `references/profiling.md` for
profiler-based diagnosis.

### Step 3 — Confirm with a profiler and counters

- Use the PyTorch profiler / trace to see whether GPU kernels are back-to-back (good) or
  have gaps where the CPU is the bottleneck (overhead).
- `nvidia-smi` GPU-Util roughly reports the percentage of sampled time when at least one
  kernel ran. Its memory-usage metric reflects capacity relevant to OOM, not bandwidth.
- For decisive kernel diagnosis, use Nsight Compute or an equivalent profiler to compare
  compute-pipeline and DRAM throughput. Classify each hotspot, then quantify its share
  of end-to-end time.

### Step 4 — Change one thing and prove the gain

- Apply one regime-matched change at a time.
- Re-run the identical benchmark and include compile time separately.
- Check correctness with tolerances appropriate to the dtype and confirm peak memory.
- Keep the change only if the target metric improves without unacceptable regressions.

## Fix Table (apply the fix that matches the regime)

| Regime | Plausible fixes |
|--------|-----------------|
| **Overhead-bound** | Tracing / graph capture (`torch.compile`, `jit.trace`, `FX`, `jax.jit`), CUDA Graphs, operator fusion, avoid tiny-tensor Python loops, move hot paths out of Python, larger batch to amortize |
| **Bandwidth-bound** | Operator fusion, fewer/larger kernels, mixed precision (fewer bytes moved), custom fused kernels (Triton), and activation checkpointing/rematerialization in selected bandwidth-limited graphs |
| **Compute-bound** | Use Tensor Cores (matmul-shaped ops with hardware- and dtype-friendly dimensions, often multiples of 8 or 16), lower precision (bf16/fp16/fp8), better matmul tiling, a cheaper algorithm, or more/faster hardware |

See `references/optimization-recipes.md` for concrete PyTorch code patterns for each fix.

## Key Optimization Concepts

- **Operator fusion**: Execute multiple operations while data remains in fast SRAM instead
  of writing intermediates to global memory. A memory-bound `x.cos().cos()` can approach
  2× speedup by halving global-memory accesses. Fused chains may cost near a single
  operation when memory traffic dominates. Cross-operator fusion generally requires
  graph capture, compilation (`torch.compile`/Inductor, NVFuser, XLA), or a manually
  fused kernel such as Triton.

- **Rematerialization / activation checkpointing**: This normally exchanges extra
  recomputation for lower saved-activation memory and can be slower. In selected fused
  or bandwidth-limited graphs it may also reduce traffic and runtime, but that outcome
  must be measured rather than assumed.

- **Compute intensity**: FLOPs per byte transferred. Raise it to move from memory-bound
  toward compute-bound. Fusion and larger tiles can help.

- **Roofline reasoning**: A tiny worked example — an A100 has ~1.5 TB/s bandwidth and
  ~19.5 TFLOPS. Its ridge point is roughly 13 FLOPs/byte, so an fp32 operation that reads
  and writes one element needs about 100 FLOPs per element to become compute-bound. See
  `references/roofline.md`.

## Agent Working Rules

1. **Measure the real workload.** Inspect the existing training or inference entry point
   and benchmark harness first, then add the smallest context-specific measurement. Run
   it when the environment has the required GPU access; otherwise ask for the smallest
   command and output needed. Do not infer end-to-end performance from a surrogate
   workload; use microbenchmarks only to isolate diagnosed hotspots.
2. **Never optimize blind.** Run the Step 0–4 diagnosis (or ask the user for profiler
   output / batch-scaling numbers) before recommending a fix.
3. **Match fix to regime** using the Fix Table. Explicitly reject fixes that target the
   wrong regime (e.g. "rewriting the loop in C++ won't help a compute-bound matmul";
   "more compute throughput alone may not help a memory-bound pointwise chain").
4. **State the diagnosis**, supporting evidence, and confidence before recommending a
   change.
5. **Prefer measurement over intuition.** When in doubt, propose a benchmark that
   isolates the variable (vary batch size, vary `repeat`/compute intensity, profile).
6. **Consider `torch.compile` as a measured candidate**, not a universal default, once
   the diagnosis is overhead-bound or fusible bandwidth-bound eager PyTorch. Preserve an
   eager baseline, then benchmark the compiled version against it. Inspect graph breaks
   and recompilations, and separate cold-start compile cost from steady state;
   compilation cost and dynamic shapes can erase the gains.
7. **Return evidence.** Use `references/diagnostic-report.md` so every recommendation
   includes baseline, observations, diagnosis confidence, patch where applicable,
   correctness check, and before/after numbers.

## Reference Files

- `references/profiling.md` — how to profile (PyTorch profiler, nvidia-smi, traces) and
  interpret results.
- `references/flop-counting.md` — counting FLOPs and measuring achieved-vs-peak FLOPS.
- `references/roofline.md` — roofline model and worked bandwidth-vs-compute arithmetic.
- `references/optimization-recipes.md` — concrete PyTorch/Triton code patterns per regime.
- `references/diagnostic-report.md` — required evidence and reporting template.

## Source

Horace He, "Making Deep Learning Go Brrrr From First Principles" (2022).
https://horace.io/brrr_intro.html
