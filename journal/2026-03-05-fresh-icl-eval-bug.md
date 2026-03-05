# 2026-03-05 - Fix broken batch-level rule match metrics for depth3_fresh_icl

## Context

Experiment 10 (`experiment/remote/10_fresh_rule_split/run.py`) evaluates models
on the `depth3_fresh_icl` task split, where every eval example uses fresh
layer-0 predicates (e.g., `r_ab3f`) generated per-example via
`build_fresh_layer0_bank`.

## Bug

`_evaluate_role_for_demo` computed batch-level rule match metrics using
`make_ar_metrics_fn(rule_bank=BASE_BANK)`. Because `BASE_BANK` does not contain
any fresh predicates, `_match_instantiated_rule` always failed, and
`_any_rule_schema_matches` classified everything as `wrong_rule_error` -- even
perfectly correct predictions.

### What was affected

Batch-level rule match metrics from `make_ar_metrics_fn`:
- `n_rule_examples`, `n_valid_rule`, `n_invalid_rule`, `n_correct_rule`
- `n_decode_error`, `n_unknown_rule_error`, `n_wrong_rule_error`
- `valid_rule_rate`, `invalid_rule_rate`, `correct_rule_rate`
- `correct_given_valid_rate`, `decode_error_rate`
- `unknown_rule_error_rate`, `wrong_rule_error_rate`

### What was NOT affected

- **Light metrics** (loss, token_acc, final_token_acc, seq_exact_acc) -- these
  compare raw token predictions against labels, no rule bank lookup needed.
- **Rollout metrics** (rollout_success_rate, rollout_*_error_rate, etc.) --
  these correctly use a per-example `temp_bank` built from fresh predicates,
  so rollout `wrong_rule_error` is genuine signal, not a bug.

## Fix

- Replaced `make_ar_metrics_fn(...)` call with `make_ar_light_metrics_fn()` in
  `_evaluate_role_for_demo`.
- Deleted the now-dead `make_ar_metrics_fn` function.
- Removed unused imports (`evaluate_rule_matches`, `extract_ar_rule_match_inputs`,
  `summarize_rule_match_metrics`).

## Note

Experiment 11 (`experiment/remote/11_fresh_rule_split_adv/run.py`) has the same
pattern and the same bug. Not fixed here since it was scoped to experiment 10.
