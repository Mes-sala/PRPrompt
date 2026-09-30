# DHBP Project Notes

## Goal

Build Dual-Harmonic Boundary Prompt (DHBP) inside ProG-V2 as the replacement
mainline candidate after R3Prompt's response/bypass evidence turned negative.
No environment installation is part of this work.

## Current Scope

- P0 only: NodeTask, frozen encoder, input-space residual prompt.
- Boundary source: few-shot train nodes from the current fold.
- Control fields: iterative normalized L and Q harmonic updates.
- Routing: local feature plus degree router over L/Q fields.
- Safety: zero-initialized `tanh` injection gate gives an exact no-op start;
  `max_intervention` caps the prompt norm relative to the input feature norm.
- Diagnostics: `dhbp_r_l`, `dhbp_r_q`, `dhbp_intervention_ratio`,
  `dhbp_injection_scale`.

## Progress Log

### 2026-09-04

- Read `DHBP 主线优化与替换决策研究报告.md` and extracted the P0 engineering
  constraints: strict identity fallback, frozen backbone, local L/Q routing,
  normalized numerical updates, and first-class intervention diagnostics.
- Inspected `R3Prompt/R3Prompt_mainline.tex`; reused its implementation lesson
  of residual control with an exact identity point, while avoiding the failed
  response-bank and edge-level bypass primitives.
- Inspected ProG-V2 strategy registration, prompt modules, NodeTask branches,
  and registry tests.
- Added `prompt_graph.prompt.DHBP.DHBPPrompt`:
  few-shot boundary bases, L/Q iterative field solving, learned local router,
  residual no-op gate, intervention cap, and diagnostics.
- Added `prompt_graph.tasker.strategies.dhbp.DHBPStrategy` and registered
  `prompt_type="DHBP"` for NodeTask.
- Wired `DHBP` into `BaseTask.initialize_prompt`,
  `BaseTask.initialize_optimizer`, `NodeTask.run`, public README strategy list,
  registry coverage tests, and NodeTask benchmark prompt lists.
- Added tests for DHBP registry presence, exact identity initialization, and a
  tiny Cora NodeTask smoke path.
- Validation performed without installing dependencies:
  `python3 -m py_compile prompt_graph/prompt/DHBP.py prompt_graph/tasker/strategies/dhbp.py`
  passed.
- Functional pytest validation is pending because this host lacks both
  `pytest` and `torch` in the available Python environment.
- Local GPU0 Cora few-shot results supplied by the user show a clear negative
  result for the current DHBP P0 implementation. Across the 21 reported
  `(shot, pretrain)` cells, DHBP never simultaneously beats GPF-plus and
  Gprompt. Best DHBP is GraphCL+DHBP at 0.3084 / 0.4631 / 0.4886 for
  1/3/5-shot, still below the strongest same-condition GPF-plus/Gprompt
  results, and below the public `None+None` full-finetune anchor at 5-shot.
- Interpretation: current DHBP is sensitive to encoder quality, because the
  pretrain ranking is preserved, but the harmonic boundary prompt has a much
  lower ceiling than strong feature/prototype prompts on Cora.

## Ablation Decision

Construct a narrow diagnostic ablation before adding P1 modules. The goal is to
identify whether the failure comes from the boundary source, L/Q field solver,
router, injection interface, or intervention budget.

Stop DHBP as a mainline candidate if the best P0 ablation still cannot beat
`frozen+head` and at least one of `GPF-plus` / `Gprompt` on GraphCL+Cora under
the same split and tuning budget.

## Next Validation

- Run targeted ablations:
  `L-only`, `Q-only`, fixed L/Q mix, no router, support-feature boundary
  prototypes, input-vs-hidden injection, and `mu/num_iters/max_intervention`
  sweeps.
- Always log `dhbp_r_l`, `dhbp_r_q`, `dhbp_intervention_ratio`, and
  `dhbp_injection_scale`; accuracy alone is not enough to interpret this model.
