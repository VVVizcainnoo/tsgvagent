# Runner / Skill Upgrade Plan

This document records the current MVP upgrade path before running the full
Charades-STA evaluation.

## Current Workflow

The agent is still training-free. It is a rule + API workflow:

1. Step1 turns each query into a state program.
2. Step2 runs sparse VLM grounding with a normal pass and, when routed, a hard
   pass.
3. Step3 verifies and repairs the candidate interval.
4. Runner scripts merge shards, export failed examples, rerun errors, and report
   metrics.

## Current 50-Video Result

Formal auto-router result after error rerun:

| Setting | Queries | mIoU | R@0.3 | R@0.5 | R@0.7 | Errors | Hard Triggered | Hard Selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| auto-router + error rerun | 125 | 0.4179 | 0.6240 | 0.4640 | 0.1920 | 5 | 98 | 22 |

This is the current automatic MVP baseline. It is useful, but it should not be
treated as the final full-run setup because the hard-mode trigger rate is too
high.

## Runner Pieces

- `scripts/write_batch_plan.py`: writes a reproducible sharded run script.
- `scripts/merge_agent_outputs.py`: merges shard outputs, overlays reruns,
  exports an error queue, and reports mIoU / recall metrics.
- `scripts/tune_router_offline.py`: tests normal-vs-hard routing margins using
  stored candidates without calling the VLM again.

## Next Upgrade

Before full-scale evaluation, improve the runner and router:

1. Make error rerun a standard loop:
   `run shards -> merge -> export errors -> rerun errors -> final merge`.
2. Tune router offline so hard mode triggers on about 20%-40% of examples.
3. Re-run the 50-video subset from scratch with the tuned no-GT router.
4. If mIoU is close to or above 0.45 with low error count, expand to a larger
   subset.

## Skill Decomposition

The code can be split into lightweight skills without introducing LangChain:

- `ProgramInductionSkill`: query to state program.
- `FrameSamplerSkill`: coarse grid and candidate-window frame selection.
- `GroundingSkill`: VLM state prediction on selected frames.
- `HardSearchSkill`: dense search only for uncertain cases.
- `VerifierSkill`: consistency and interval repair.
- `RouterSkill`: choose normal or hard path from confidence signals.
- `RunnerSkill`: shard, retry, merge, and evaluate batches.
- `AnalyzerSkill`: failure grouping and next-change suggestions.

The immediate priority is `RunnerSkill` and `RouterSkill`; the rest can stay as
plain Python modules until the full evaluation design stabilizes.
