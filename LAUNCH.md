# Launch Playbook

Publishing a correct skill is not enough. Earn attention with proof, make installation
frictionless, and distribute where Agent Skills users already browse.

## Before launch

- Publish under the repository name `gpu-brrr` (memorable brand); keep the discovery
  keywords in the skill `description` and GitHub topics, not the slug.
- Add GitHub topics: `agent-skills`, `pytorch`, `gpu`, `cuda`, `torch-compile`,
  `performance`, `claude-code`, `opencode`, and `hermes-agent`.
- Run the skill against two public workloads. Preserve the full agent transcript,
  profiler screenshot, reproduction command, correctness result, and before/after
  metrics. Do not cherry-pick only successful cases.
- Turn the strongest case into a 30–60 second terminal recording or short GIF:
  bad generic advice, measured diagnosis, one patch, verified result.
- Create a tagged GitHub release with the standalone skill archive attached.

## Positioning

Lead with the failure mode, not the format:

> GPU Brrr: stop letting coding agents guess at GPU optimization. This Agent Skill makes
> them prove whether a workload is compute-, bandwidth-, or overhead-bound before editing.

Support that hook with one real result:

> On WORKLOAD and HARDWARE, the agent rejected ASSUMPTION, found BOTTLENECK, changed
> ONE THING, and improved TARGET METRIC from X to Y with correctness preserved.

Replace every placeholder with reproducible data. Never market an unmeasured speedup.

## Distribution

1. Submit the repository to
   [Awesome Agent Skills](https://github.com/VoltAgent/awesome-agent-skills).
2. Make the skill installable through the Hermes ecosystem and consider submitting it
   to [HermesHub](https://github.com/amanning3390/hermeshub).
3. Link the standard and compatibility:
   [Agent Skills](https://agentskills.io/),
   [Claude Code](https://docs.claude.com/en/docs/claude-code/overview),
   [OpenCode](https://opencode.ai/docs/skills/), and
   [Hermes Agent](https://github.com/NousResearch/hermes-agent).
   For Claude Code, mention both install paths: native skill copy and the optional
   `/plugin marketplace add` + `/plugin install gpu-brrr@gpu-brrr` flow.
4. Share the measured case study in PyTorch, CUDA, ML systems, OpenCode, and Hermes
   communities where self-promotion is allowed. Adapt the explanation to each audience.
5. Ask practitioners for adversarial test cases rather than asking only for stars.

## Launch posts

### Short post

```text
Most AI coding agents optimize GPU code by guessing:
"use torch.compile", "increase batch size", "write Triton".

I made an open Agent Skill that forces them to measure first:
1. establish a correct baseline
2. classify compute vs bandwidth vs overhead
3. change one thing
4. verify outputs and report before/after evidence

Demo: [measured result]
Install: [repository URL]

Give it a workload that fools it. Issues and counterexamples are welcome.
```

### Hacker News title

```text
Show HN: A skill that makes coding agents diagnose GPU bottlenecks before optimizing
```

Avoid claiming affiliation with Horace He or PyTorch. Credit the foundational article
prominently and describe this project as an independent operationalization.

## After launch

- Respond quickly to installation failures and technical counterexamples.
- Label reproducible benchmark reports as case studies.
- Publish a small evaluation set that checks whether agents avoid common false
  diagnoses. Track results across agent clients and models.
- Release framework references incrementally rather than bloating the core skill.
- Keep a changelog and ship small tagged releases.

## Success metrics

Do not optimize only for stars. Track:

- Successful installs and copy-paste installation failures.
- Number of independent reproduced case studies.
- Issues that reveal wrong or overconfident diagnoses.
- External contributors and registry inclusions.
- Returning users and downstream repositories referencing the skill.
