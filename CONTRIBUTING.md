# Contributing

Thank you for helping make GPU performance advice more measurable and reliable.

## Pull request requirements

Changes to technical recommendations should include:

1. The workload, hardware, software versions, shapes, and dtype.
2. A reproducible benchmark with warm-up and correct synchronization.
3. Cold-start and steady-state numbers separated where compilation is involved.
4. Correctness checks and peak-memory impact.
5. Profiler or hardware-counter evidence for a claimed bottleneck.
6. Links to authoritative documentation for API or hardware claims.

Do not add:

- Universal thresholds presented as laws.
- Speedup claims without a baseline and reproduction steps.
- Advice that treats `GPU-Util` or low FLOPS as a complete diagnosis.
- Large copied sections from third-party articles or documentation.

## Skill quality

Keep `SKILL.md` concise and put detailed material in `references/`. Preserve the
diagnose, intervene, verify workflow. Validate the skill directory with an
Agent Skills-compatible validator before submitting.

## Commit style

Use concise imperative subjects, for example:

```text
Clarify checkpointing trade-offs
Add ROCm profiling reference
Test graph-break diagnosis prompt
```
