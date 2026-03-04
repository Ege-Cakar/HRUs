# 2026-02-26 - FOL Layer Sparsity Sensitivity Reproduction

## Context
Goal for this iteration was to ground "realistic" `task/layer_fol.py` parameter choices in measured behavior, especially around sparsity and long-distance feasibility for function-application-like reasoning.

A key question was which existing knobs naturally control:
- how many viable paths exist,
- how long feasible paths remain,
- whether large constant vocabularies by themselves make the task sparse.

Backbone/maze generation is intentionally deferred to a follow-up iteration.

## Reproduction Protocol
Implemented a new sensitivity study in `task/layer_fol_stats.py`.

Base config used for reproduction:
- `n_layers=30`
- `predicates_per_layer=16`
- `rules_per_transition=32`
- `arity_max=3`
- `vars_per_rule_max=4`
- `k_in_max=3`
- `k_out_max=3`
- `constants=c0..c63` (64 constants)
- `initial_ant_max=3`
- `sample_max_attempts=2048`
- `max_unify_solutions=128`

Feasibility metric:
- target distance: `12`
- seeds: `[0,1,2,3,4]`
- trials per seed: `12`
- score: fraction of successful `sample_fol_problem(...)` calls.

## Results
Mean feasible rate by sweep value:

### `rules_per_transition`
- `8 -> 0.000`
- `12 -> 0.000`
- `16 -> 0.000`
- `24 -> 0.683`
- `32 -> 1.000`
- `48 -> 1.000`

### `k_in_max`
- `1 -> 1.000`
- `2 -> 1.000`
- `3 -> 1.000`
- `4 -> 0.400`
- `5 -> 0.083`

### `k_out_max`
- `1 -> 0.000`
- `2 -> 0.850`
- `3 -> 1.000`
- `4 -> 1.000`

### `predicates_per_layer`
- `8 -> 1.000`
- `12 -> 1.000`
- `16 -> 1.000`
- `24 -> 0.517`
- `32 -> 0.200`

### `constants_count`
- `4 -> 1.000`
- `16 -> 1.000`
- `64 -> 1.000`
- `256 -> 1.000`

## Interpretation
Primary sparsity/feasibility controls in the current random layered generator are:
1. `rules_per_transition`
2. `predicates_per_layer`
3. `k_in_max`
4. `k_out_max`

In contrast, constant-pool size mainly affects lexical/entity diversity (vocab and grounding variety), with weak direct effect on long-distance sampling feasibility in this regime.

The results show phase-transition-like behavior: some knobs move quickly from near-impossible to near-certain feasibility. This suggests parameter selection should target specific feasibility bands, not arbitrary monotone scaling.

## Practical Balanced Defaults
For realistic but trainable long-horizon behavior (online sampling):
- `n_layers=96`
- `predicates_per_layer=32`
- `rules_per_transition=48`
- `k_in_max=3`
- `k_out_max=3`
- `arity_max=3`
- `vars_per_rule_max=6`
- `constants=512`
- train distances: `1..24`
- eval distances: up to `64`

## Deferred: Backbone/Maze Path Injection
Not implemented in this iteration.

Design target for next iteration:
- Add a backbone generator that guarantees at least one sparse path of desired length.
- Keep distractor rules and optional dead-end branches to preserve realistic decision pressure.
- Proposed controls:
  - `backbone_length`
  - `branch_factor`
  - `dead_end_depth`
  - `backbone_count`

Expected outcome:
- explicit control over path length and branching,
- reduced dependence on fragile phase-boundary tuning,
- cleaner sparse-long-path OOD studies.

## Next Steps
1. Add `rule_bank_mode={random,backbone}` with a backbone builder in `fol_rule_bank.py`.
2. Extend `layer_fol_stats.py` to compare random vs backbone banks under matched difficulty budgets.
3. Add tests asserting guaranteed backbone reachability and controlled branching.
