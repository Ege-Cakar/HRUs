# 2026-02-24 - Layer Task Tool-Use Continuum

## Task Summary
Capture a concrete map between the current LayerTask family and realistic tool
use settings (BFCL-like and beyond), while preserving simplicity, control, and
tractability for mechanistic and scaling studies.

## Why LayerTask Already Looks Like Tool Use
The current LayerTask abstraction is already close to a "next tool call" policy:
- prompt: current known state (antecedents) + goal
- target: one valid transition rule that moves state toward the goal
- rollout: latent sequence of decisions (distance steps), even when training on
  a sampled single step

This corresponds to:
- state -> available tools -> choose tool -> state update

In `FOLLayerTask`, the match is even tighter:
- selecting an instantiated rule template is analogous to selecting a tool plus
  arguments (via substitutions/unification)

## Where It Is Simpler Than BFCL/Real Tool Use
Compared with BFCL and production tool agents, LayerTask currently omits:
- abstention (`NO_TOOL` / ask-clarification / cannot-solve)
- non-executable or inapplicable call outcomes in the supervision target
- schema/docs noise and near-miss tools
- multi-turn user dialogue as context
- post-call observation uncertainty and error recovery loops
- policy constraints (ordering, permissions, safety gates)

These omissions are good for control, but they define the realism gap.

## Minimal Extensions That Preserve Tractability
The most useful low-complexity additions are orthogonal "knobs" rather than a
full environment rewrite.

1) `NO_TOOL` branch
- Add examples where no transition is applicable or sufficient.
- Forces calibrated abstention behavior and reduces over-calling.

2) Tool schema conditioning
- Represent each rule with a compact schema block (name, args, precondition
  summary, effect summary), plus distractor schemas from same layer.
- Keeps symbolic ground truth while introducing retrieval pressure.

3) Typed argument prediction
- Move target from `rule_text` to `(rule_id, argument_binding)` in FOL setting.
- Can be evaluated by exact match and execution validity.

4) Executable outcomes
- Given a predicted call, deterministically emit result token classes:
  `success`, `inapplicable`, `bad_args`, `policy_blocked`.
- Enables recovery-style training while remaining synthetic and deterministic.

5) Short-horizon multi-step traces
- Supervise 2-4 step episodes with observation append after each step.
- Keeps search shallow and analyzable, but introduces compounding errors.

6) Lightweight policy constraints
- Add constraints like "verify before modify" or "read-only tools first."
- Distinguishes capability from policy compliance, mirroring real agents.

## Continuum: LayerTask -> BFCL-like -> Full Agentic
### Stage 0: Current LayerTask
- target: next valid rule statement
- environment: deterministic symbolic state
- eval: rule-level exact match / validity

### Stage 1: LayerTask + Abstention
- add `NO_TOOL` and impossible/underspecified prompts
- eval: selective prediction quality (precision-recall of calling)

### Stage 2: FOLLayer as Typed Calling
- target: `(rule_template, substitution)`
- eval: exact-call match + executable correctness

### Stage 3: BFCL-like Single-Turn Calling
- include tool schemas/docs, distractor tools, argument formatting constraints
- keep one-turn input, one-call output
- eval: call format, tool selection, arg correctness

### Stage 4: BFCL-like Multi-Turn with Execution Feedback
- append deterministic tool outputs/errors to context
- allow corrective second call
- eval: task success over short trajectories

### Stage 5: Policy-Constrained Tool World
- add permissions/order constraints and explicit failure types
- evaluate capability vs policy adherence separately

### Stage 6: Full Agentic Environment
- partial observability, noisy external returns, long-horizon planning, and
  non-stationary APIs

This gives a natural research ladder: each stage adds one major difficulty
dimension while keeping earlier invariants intact.

## Suggested Canonical Data Contract (for Stages 2-5)
Keep a stable internal record schema so models and analysis code do not churn:
- `context`: user/request text + state serialization + optional history
- `tools`: list of tool/rule schemas available at this step
- `target_call`: `tool_id` + structured args OR `NO_TOOL`
- `exec_result`: deterministic result class + optional state delta text
- `meta`: difficulty tags (`distance`, distractor_count, policy_flags)

A stable contract is the key to ablations and transfer studies.

## Evaluation Decomposition
For comparability with BFCL-style analysis, separate metrics:
- call decision: call vs abstain
- tool selection: correct tool id
- argument binding: exact args / normalized args
- execution validity: callable with current state/policy
- trajectory success: goal reached within step budget

This decomposition helps diagnose whether failures are retrieval, reasoning,
formatting, or control-policy failures.

## Immediate, Practical Next Dataset Variant
A pragmatic "next" benchmark that stays simple:
- Base: `FOLLayerTask`
- Add: `NO_TOOL`, distractor schemas, and explicit result classes
- Keep: deterministic transitions and short horizons (<=4 steps)
- Output: one-step and two-step versions for controlled scaling

This should be substantially closer to BFCL semantics without sacrificing the
synthetic control needed for mechanistic experiments.

