# Diagnostic Report Template

The agent must return this evidence before claiming an optimization succeeded.

## Target

- Workload and mode:
- Hardware and software:
- Shapes, batch size, sequence length, dtype:
- Primary metric:
- Correctness tolerance:

## Benchmark protocol

- Warm-up iterations:
- Measured iterations:
- Timing method:
- Cold-start/compile time:
- Steady-state median:
- p95 or IQR:
- Peak allocated/reserved memory:

## Evidence

| Observation | Measurement | Interpretation |
|---|---:|---|
| Batch/shape scaling | | |
| GPU timeline gaps | | |
| Top kernels and time share | | |
| Achieved FLOPS vs matching peak | | |
| DRAM throughput | | |
| Compute-pipeline throughput | | |
| Graph breaks/recompiles | | |
| Input/communication/sync time | | |

## Diagnosis

- Dominant regime:
- Hotspot scope: whole step or named kernels
- Confidence: low / medium / high
- Alternative explanations not yet ruled out:

## Intervention

- Single change:
- Why it matches the evidence:
- Expected trade-offs:

## Result

| Metric | Baseline | Candidate | Change |
|---|---:|---:|---:|
| Steady-state median | | | |
| p95 or IQR | | | |
| Throughput | | | |
| Peak memory | | | |
| Compile/cold-start time | | | |

- Correctness check:
- Decision: keep / reject / investigate
- Reproduction command:
