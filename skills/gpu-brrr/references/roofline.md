# The Roofline Model & Bandwidth Arithmetic

The roofline model ties compute and memory bandwidth into one picture, so you can tell
at a glance whether an operation *can* be compute-bound.

## Arithmetic (compute) intensity

```
arithmetic_intensity = FLOPs performed / bytes moved to/from DRAM   [FLOP/byte]
```

- Low intensity (few FLOP per byte) → memory-bound.
- High intensity (many FLOP per byte) → compute-bound.

## The roofline

```
attainable_FLOPS = min( peak_FLOPS,  peak_bandwidth * arithmetic_intensity )
```

- On the **sloped** part (`bandwidth * intensity`), you're memory-bound: performance
  rises with intensity.
- On the **flat** part (`peak_FLOPS`), you're compute-bound: more intensity doesn't help.
- The "ridge point" intensity = `peak_FLOPS / peak_bandwidth` is the minimum intensity
  needed to be compute-bound.

## Worked example (A100, from the article)

- Peak bandwidth ≈ 1.5 TB/s.
- General (non-tensor-core) FLOPS ≈ 19.5 TFLOPS.
- fp32 = 4 bytes/element.

In the time the GPU does ~20 trillion ops, it can only move ~400 billion fp32 numbers
(1.5e12 / 4). A unary op like `x * 2` must **read** the tensor and **write** it back.
So per element you pay ~2 memory transactions but do ~1 FLOP → intensity far below the
ridge point. You must fold on the order of **~100 ops per element** before compute, not
bandwidth, dominates. This is exactly why isolated pointwise ops (and unfused chains of
them) are memory-bound, and why **fusion** — doing many ops per load — is the fix.

## Ridge point intuition

Ridge point ≈ 19.5e12 / 1.5e12 ≈ **13 FLOP/byte** on A100 (general FLOPS). For fp32
(4 B) that's ~52 FLOP per element loaded. Anything doing less math per byte than the
ridge point is memory-bound and cannot be sped up by a faster-at-compute GPU.

## How to raise arithmetic intensity

- **Operator fusion**: reuse data in SRAM across many ops instead of round-tripping DRAM.
- **Larger matmul tiles / bigger batch**: more reuse of loaded operands.
- **Lower precision** (bf16/fp16/fp8): fewer bytes per element → more FLOP/byte for the
  same math, plus Tensor Core eligibility.
- **Rematerialization**: recompute instead of loading a saved activation when compute is
  cheaper than the memory round-trip.
