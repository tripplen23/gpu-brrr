# Optimization Recipes (PyTorch)

Concrete patterns, grouped by the regime they fix. Apply only after diagnosis.

## Overhead-bound fixes

Symptom: runtime flat vs batch size; profiler shows GPU gaps while CPU is busy;
low nvidia-smi GPU-Util; many tiny fast kernels.

### torch.compile (first thing to try)
```python
compiled_model = torch.compile(model)
# Optional after baseline: torch.compile(model, mode="max-autotune")
```

Measure compile time separately. Check graph breaks and recompilations; dynamic shapes
and unsupported Python can fragment the graph and remove the expected benefit.

### CUDA Graphs (eliminate per-launch overhead for static shapes)
```python
# via torch.compile:
model = torch.compile(model, mode="reduce-overhead")   # uses CUDA graphs internally
```

### Increase batch size
Amortizes fixed per-step overhead over more work; often the simplest win.

### Avoid tiny-tensor Python loops / .item() / .cpu() syncs
- Replace Python loops over small tensors with vectorized ops.
- Remove needless `.item()`, `.cpu()`, `print(tensor)` inside the hot loop — each forces
  a device sync and stalls the "run ahead" that hides overhead.

## Bandwidth-bound fixes

Symptom: runtime scales with size; achieved FLOPS is a few % of peak; profiler dominated
by pointwise/normalization/elementwise kernels.

### Operator fusion (the key optimization)
```python
# Let the compiler fuse pointwise chains, norms, activations, biases, etc.
model = torch.compile(model)
```
Fusing `a.cos().cos()` (or add+relu, layernorm, softmax) into one kernel removes DRAM
round-trips. A fused chain costs ~the same as a single op.

### Mixed precision (fewer bytes moved + Tensor Cores)
```python
scaler = torch.amp.GradScaler("cuda")
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    out = model(x); loss = loss_fn(out, y)
```

### Activation checkpointing / rematerialization
```python
from torch.utils.checkpoint import checkpoint
out = checkpoint(block, x, use_reentrant=False)
```
Checkpointing typically lowers activation memory by recomputing during backward and
therefore often increases step time. It can occasionally improve a bandwidth-limited
or fused graph; benchmark rather than promising a speedup.

### Custom fused kernels (max effort)
Write a Triton kernel to fuse an op pattern the compiler can't:
```python
import triton, triton.language as tl
# fuse e.g. bias + activation + dropout in one pass over DRAM
```
Any two adjacent PyTorch ops are a fusion opportunity. Also: NVFuser, XLA for auto-fusion.

## Compute-bound fixes

Symptom: achieved FLOPS near peak (≥~70–80%); profiler dominated by large matmul/gemm/conv.

### Use Tensor Cores
- Use bf16/fp16 (or TF32 for fp32 matmuls): `torch.set_float32_matmul_precision("high")`.
- Make matmul dims multiples of 8 (fp16) / 16 to hit Tensor Core paths.
- Pad vocab / hidden dims to friendly multiples.

### Lower precision further
- fp8 on Hopper+ (via transformer-engine) for the largest matmuls.

### Cheaper algorithm
- FlashAttention instead of naive attention (also fuses → less memory traffic).
- Lower-rank / sparse / quantized variants (e.g. QLoRA for fine-tuning).

### Hardware
- If genuinely compute-bound at high utilization, the remaining lever is a
  faster/larger accelerator ("give Nvidia more money").

## Quick reference

| Diagnosis | First fix | Then |
|-----------|-----------|------|
| Overhead-bound | Preserve baseline, then `torch.compile` | CUDA graphs, bigger batch, remove syncs |
| Bandwidth-bound | Confirm counters, then compile/fuse | mixed precision, layout, Triton |
| Compute-bound | Tensor Cores + bf16/TF32 | fp8, better algorithm, more hardware |
