# 2026-02-05 - Normative OOD Pipeline Dry Run

## Motivation
- The project goal is to make neural reasoning behavior predictable on a controlled propositional logic task, not just to report raw accuracy.
- Bag-of-words style baselines were too weak, while small Transformers appeared to learn complex patterns; we needed stronger normative comparators and a more publishable framing.
- We want an analysis stack that can answer: given training split + architecture, how well should OOD generalization be expected to transfer?

## Purpose of Today's Work
- Implement and validate a full experiment path for comparing Transformer performance against compact normative models on implication-size OOD shifts.
- Ensure the pipeline logs consistent metrics across model families, including calibration and NLL, so we can build a predictive meta-model of OOD performance.
- Confirm the end-to-end workflow runs locally in a small test configuration before larger remote sweeps.

## What Was Implemented
- Added `model/normative/` package:
  - `proof_features.py`: symbolic/proof-state feature extraction and candidate-choice dataset construction.
  - `loglinear.py`: log-linear choice policy baseline.
  - `kernel.py`: kernel retrieval baseline.
  - `bounded_prover.py`: hand-designed bounded-rational heuristic policy.
  - `eval.py`: shared evaluation (`rule_acc`, `pos_acc`, `joint_acc`, `rule_membership_acc`, `nll`, `calibration_ece`).
- Added common helpers in `common.py`:
  - `multiclass_nll_from_logits`
  - `expected_calibration_error_from_logits`
- Added remote sweep:
  - `experiment/remote/4_normative_ood/run.py`
  - `experiment/remote/4_normative_ood/sb_4_normative_ood.sh`
- Added analysis script:
  - `experiment/4_normative_ood.py`
- Added tests:
  - `tests/test_normative.py`

## Dry Run Configuration and Result
- Enabled `TEST CONFIGS` in `experiment/remote/4_normative_ood/run.py`:
  - `TRAIN_MAXIMA=[5]`, `TEST_MAX=10`, single architecture case, `TRAIN_ITERS=200`, `RUN_SPLIT=1`.
  - Local dataset path: `task/prop_gen/data/toy_imply`.
- Ran: `python run.py 0` in `experiment/remote/4_normative_ood`.
- Outcome:
  - `TOTAL CASES: 1` and successful completion.
  - Training checkpoints:
    - iter 100: `train_joint=0.9531`, `test_joint=0.7500`, `test_nll=0.6167`
    - iter 200: `train_joint=0.9531`, `test_joint=0.8125`, `test_nll=0.4393`
  - Normative sample counts: train `512`, test `256`.
  - Output file: `experiment/remote/4_normative_ood/set/res.797919920.pkl`
  - Output schema validated: 4 rows (`transformer`, `loglinear`, `kernel`, `bounded_prover`) with expected metrics.

## Analysis Output Check
- Ran `experiment/4_normative_ood.py` on dry-run output.
- Produced:
  - `fig/4_normative_ood/ood_joint_by_family.svg`
  - `fig/4_normative_ood/calibration_nll_by_family.svg`
  - `fig/4_normative_ood/summary.csv`
- Expected warning observed: not enough rows for ridge predictor fit in test mode.

## Notes for Future Me
- CUDA plugin warnings appeared in this environment; run fell back to CPU and completed correctly.
- Next meaningful step is reverting to full sweep settings and generating enough runs for the predictive meta-model and regime-boundary analysis.
