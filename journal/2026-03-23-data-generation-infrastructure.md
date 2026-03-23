# 2026-03-23 — Data Generation Infrastructure

## Summary

Built the complete FOL data generation pipeline for the recursive transformer study. Three task conditions are now supported: **internalized rules**, **pure ICL**, and **hybrid ICL** (internalized + in-context). All conditions share a single deterministic rule bank. Both online (streaming) and offline (ArrayRecord shards) generation paths work. 194 tests pass across all FOL modules.

## Research Context

The project compares four architecture families — Standard Transformer, Recursive Transformer, Hybrid (GDN + Attention, 3:1), and Hybrid Recursive — on synthetic FOL reasoning tasks that scale from ~10M to ~600M parameters. Two axes of variation:

1. **Internalized reasoning**: Can the model compose D rule applications from memory (no rules in prompt)?
2. **Hybrid ICL**: Can the model combine internalized rules with novel in-context rules it has never seen?

The key experimental lever is **derivation depth D**. Recursive models should maintain performance at depths where standard models collapse by iterating the recurrent core beyond training depth.

## Task Conditions

### Condition 1: Internalized Rules
- Prompt: `facts ⊢ goal <START>`
- No rules in context. The model must have memorized the rule bank during training.
- Tests pure iterative reasoning capacity.
- `FOLLayerTask` with `max_n_demos=0`.

### Condition 2: Pure ICL (legacy, kept for backward compat)
- Prompt: `demo_rule_1 <SEP> demo_rule_2 <SEP> ... <SEP> facts ⊢ goal <START>`
- All rules come from context. Zipf-weighted sampling controls noise.
- Tests retrieval + application under noise.
- `FOLLayerTask` with `max_n_demos > 0`.

### Condition 3: Hybrid ICL (new, primary ICL condition)
- The model has K internalized rules AND receives novel rules in-context.
- Per-problem, each transition in the derivation chain is randomly assigned to use an internalized rule OR a fresh in-context rule (probability `p_fresh`).
- Fresh rules have novel predicates on LHS but standard predicates on RHS, enabling composability: fresh-rule outputs feed directly into internalized-rule inputs at the next step.
- Three eval sub-conditions of increasing difficulty:
  - **train**: Fresh rules from the training pool (sanity check).
  - **rule_gen**: Same fresh predicates as training, but new rule structures. Tests whether the model generalizes to new logical relationships with familiar symbols.
  - **pred_gen**: Entirely new predicates + new rules. Tests true in-context learning of novel symbols — the model has never seen these predicates in any form.

## Architecture

### Rule Bank Structure (Hybrid ICL)

```
HybridICLBank
├── base_bank: FOLRuleBank (N layers, K rules/transition — internalized)
├── train_predicates_by_layer: fresh predicates for training demos
├── eval_pred_predicates_by_layer: held-out fresh predicates (disjoint from train)
├── train_rules: fresh rules using train predicates → standard RHS
├── eval_rule_rules: fresh rules using train predicates, different structure
├── eval_pred_rules: fresh rules using eval predicates → standard RHS
└── p_fresh: probability a transition uses fresh vs internalized
```

### Composability Mechanism

Fresh rules bridge novel predicates to standard predicates:
```
Fresh rule:        r_abc(x) ∧ r_def(y) → r3_1(x, y)   [novel LHS, standard RHS]
Internalized rule: r3_1(x, y) ∧ r3_2(z) → r4_5(x, z)  [standard both sides]
```

Facts flow naturally between fresh and internalized transitions because standard predicates serve as the interface layer.

### Per-Problem Sampling

```
1. Pick start layer, generate initial facts (mix of base + fresh predicates).
2. For each step i in 0..D-1:
     coin flip with p_fresh → internalized or fresh transition.
     Select from appropriate rule pool, unify against current facts.
     Derive new facts for next layer.
     If fresh: track template rule for in-context demos.
3. Build demo sequence:
     - Required: all fresh rules used in this problem.
     - Noise: additional demos from Zipf distribution over base bank.
4. Tokenize: demos <SEP> ... <SEP> facts ⊢ goal <START>
```

## Files Created/Modified

### New Files

| File | Purpose |
|------|---------|
| `task/layer_gen/util/fol_rule_bank/_hybrid_icl.py` | `HybridICLBank` dataclass, `build_hybrid_icl_bank()`, serialization |
| `task/layer_gen/util/fol_rule_bank/_hybrid_icl_sampling.py` | `sample_hybrid_icl_problem()` with three eval modes |
| `task/layer_gen/generate_fol_dataset.py` | Offline ArrayRecord generation for all conditions |
| `tests/test_hybrid_icl.py` | 21 tests for hybrid ICL construction, sampling, composability, persistence |
| `model/eval_adapters.py` | Restored from git (AR/completion decoding adapters) |

### Modified Files

| File | Change |
|------|--------|
| `task/fol_task_factory.py` | Added `HybridICLConfig`, `HybridICLTask` iterator, `make_hybrid_icl_task()`, `make_hybrid_icl_eval_tasks()`, `from_rule_bank_path()`, mode/ds_path params, `DepthCurriculum` (linear/exponential/manual), `CurriculumTaskManager` |
| `task/layer_gen/util/fol_rule_bank/__init__.py` | Export hybrid ICL types |
| `task/layer_fol/__init__.py` | Fixed broken import (eval_adapters) |
| `tests/test_fol_task_factory.py` | Added `test_from_rule_bank_path`, fixed tokenizer test |

## Usage

### Online (Fast Iteration)

```python
from task.fol_task_factory import FOLTaskFactory, HybridICLConfig, RuleBankConfig

factory = FOLTaskFactory(rule_bank_seed=42, d_train_max=5, d_eval_max=15)

# Internalized
task_int = factory.make_internalized_task(batch_size=64)
xs, ys = next(task_int)

# Hybrid ICL — training
hcfg = HybridICLConfig(fresh_predicates_per_layer=8, fresh_rules_per_transition=8)
task_hybrid = factory.make_hybrid_icl_task(batch_size=64, eval_mode="train", hybrid_config=hcfg)
xs, ys = next(task_hybrid)

# Hybrid ICL — predicate-level generalization eval
task_eval = factory.make_hybrid_icl_task(batch_size=64, eval_mode="pred_gen", hybrid_config=hcfg)
xs, ys = next(task_eval)
```

### Offline (Reproducible Shards)

```bash
python -m task.layer_gen.generate_fol_dataset \
  --out-dir data/fol_seed42 --seed 42 \
  --conditions internalized,hybrid_icl_train,hybrid_icl_rule_gen,hybrid_icl_pred_gen \
  --min-distance 1 --max-distance 8 \
  --examples-per-distance 50000 --workers 32 \
  --fresh-predicates-per-layer 8 --fresh-rules-per-transition 8 \
  --p-fresh 0.5 --max-n-demos 16
```

Output:
```
data/fol_seed42/
  rule_bank.json
  hybrid_icl_bank.json
  internalized/distance_001/ ... /metadata.json
  hybrid_icl_train/distance_001/ ... /metadata.json
  hybrid_icl_rule_gen/distance_001/ ... /metadata.json
  hybrid_icl_pred_gen/distance_001/ ... /metadata.json
```

## Test Results

```
194 passed in 9.57s (0 failures, 0 regressions)
  - test_fol_task_factory.py:     32 tests (incl. curriculum)
  - test_hybrid_icl.py:           21 tests
  - test_layer_fol_task.py:       56 tests
  - test_fol_rule_bank.py:        19 tests
  - test_layer_fol_eval.py:       74 tests
```

## Depth Curriculum

Added `DepthCurriculum` and `CurriculumTaskManager` to support training with progressively increasing derivation depth. This is motivated by both the recursive transformer literature (Huginn, Ouro use curriculum over recurrence count) and the length generalization evaluation design: train on depths 1..D, evaluate on depths 1..2D.

### Three scheduling modes

**Linear**: Depth increases by 1 every `steps_per_depth` steps.
```python
curriculum = DepthCurriculum.linear(d_start=1, d_max=8, steps_per_depth=5000)
# step 0: D≤1, step 5000: D≤2, ..., step 35000: D≤8
```

**Exponential**: Intervals grow geometrically — spend more time at higher depths.
```python
curriculum = DepthCurriculum.exponential(d_start=1, d_max=8, steps_per_depth=1000, growth_factor=2.0)
# step 0: D≤1, step 1000: D≤2, step 3000: D≤3, step 7000: D≤4, ...
```

**Manual**: Explicit phase boundaries for full control.
```python
curriculum = DepthCurriculum.manual([(0, 1), (5000, 3), (20000, 6), (50000, 8)])
```

### CurriculumTaskManager

Wraps a factory + curriculum + pre-generated offline shards. At each training step, checks if the max depth has changed. If so, rebuilds the task with the new `distance_range` pointing at the **same `ds_path`** (which contains shards for all depths up to `d_eval_max`). No data regeneration needed — the curriculum just controls which depth range is sampled from.

```python
manager = CurriculumTaskManager(
    factory=factory,
    curriculum=DepthCurriculum.linear(d_start=1, d_max=8, steps_per_depth=5000),
    condition="internalized",
    ds_path="data/fol_seed42/internalized",
    batch_size=64,
)
for step in range(total_steps):
    xs, ys = manager.next_batch(step)
    # depth range automatically expands at phase boundaries
```

### Length generalization evaluation

The evaluation design follows directly from the curriculum:
- **In-distribution**: Eval at depths 1..D (same range as final training phase).
- **Length generalization**: Eval at depths D+1..2D (beyond training). Recursive models should degrade gracefully here because they can iterate the core extra times. Standard models should collapse because they have no mechanism to extend computation.

The per-depth eval tasks (`factory.make_internalized_eval_tasks(depths=range(1, 2*D+1))`) are already generated for all depths up to `d_eval_max` in the offline shards.

## Key Design Decisions

1. **Shared rule bank**: All conditions use the exact same base bank from the same seed. The only difference is what appears in context.

2. **Fresh predicates on LHS, standard on RHS**: This is the composability mechanism. Fresh rules produce standard-predicate facts, so internalized rules at the next step can consume them without special handling.

3. **Random per-transition mixing**: Each derivation step independently decides fresh vs internalized. This is more naturalistic than fixed-layer assignments and produces stronger evidence if the model handles it.

4. **Three eval conditions for generalization**: Rule-level (familiar symbols, new structure) and predicate-level (entirely novel symbols) test different generalization capabilities. The gap between them reveals how much the model relies on symbol familiarity vs pure in-context learning.

5. **Disjoint predicate split**: Train and eval-pred predicates are guaranteed disjoint. The model literally cannot have seen eval-pred predicates during training in any form.

6. **`HybridICLTask` as standalone iterator**: Unlike `FOLLayerTask` (which handles single-bank sampling), the hybrid task needs custom sampling logic that mixes two rule pools per-transition. Rather than overcomplicating `FOLLayerTask`, we built a clean standalone iterator.

## Open Questions / Next Steps

- **Architecture implementations**: Need to build Recursive Transformer, Hybrid (GDN+Attention), and Hybrid Recursive in Flax NNX. The existing `model/transformer.py` is the starting point.
- **Training loop integration**: The factory outputs need to plug into `train.py` (JAX path) and `train_hf.py` (HuggingFace path for Qwen fine-tuning).
- **Evaluation pipeline**: The existing `task/layer_fol/eval/` has rollout evaluation, but needs adaptation for hybrid ICL (tracking which transitions were fresh vs internalized).
- **Scaling design**: Define the 10 synthetic model sizes (~10M–600M params) and map them to the 5 Qwen sizes (0.8B–27B).
- **Ablations**: Attention ratio in recurrent core, number of recurrences vs depth D, halting/early exit.
