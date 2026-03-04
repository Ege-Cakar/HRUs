# 2026-02-23 - FOL LayerTask Generalization Plan (Brainstorm + Spec)

## Task Summary
Generalize `task/layer.py` / `task/layer_gen/generate_layer.py` from
propositional layered modus ponens to a first-order logic variant that
preserves the existing task structure (layered transitions, distance-based
rollouts, prompt/completion training format), while introducing quantification
and relational structure.

## Repo-Grounded Analysis
I inspected:
- `task/layer.py`
- `task/layer_gen/generate_layer.py`
- `task/layer_gen/util/rule_bank.py`
- `task/layer_gen/util/tokenize_layer.py`
- tests:
  - `tests/test_layer_task.py`
  - `tests/test_generate_layer.py`
  - `tests/test_layer_tokenize.py`

Key findings:
- Current task already has strong separation of concerns:
  - rule-bank sampling
  - tokenizer
  - generator to ArrayRecord
  - online/offline task loader with batching/objective controls
- Training-facing contract is simple and stable:
  - `__next__() -> (xs, ys)` for `autoregressive` or `all_at_once`
- Existing invariants worth preserving:
  - per-distance count invariant (`examples * distance` records)
  - metadata-backed global max sequence support
  - SEP/EOT conventions
  - completion validity checking against rule bank + source layer
- There is no existing first-order task implementation in `task/`.

## Conceptual Clarification
This is not just renaming propositional symbols:
- `p` -> `R(a,b)` alone is insufficient.
- First-order difficulty comes from **unification/substitution constraints**
across arguments:
  - repeated variables (e.g. `R(x,x)`) encode equality constraints
  - argument positions and variable-sharing pattern matter
- So relation name + arity are necessary but not sufficient; clause structure
carries core reasoning content.

## Decisions Locked During Planning
- Base family: **function-free Horn first-order logic** (generalized modus
ponens + unification).
- Quantifier surface: keep universal quantification implicit (`forall` by
convention) for v1 simplicity.
- Completion target: **instantiated clause text** (not schema+substitution,
not next-fact-only).

## Proposed Implementation Plan

### 1) New FOL rule-bank + sampler
Add `task/layer_gen/util/fol_rule_bank.py`:
- `FOLLayerRule`
- `FOLRuleBank`
- `FOLSampledProblem`
- `build_random_fol_rule_bank(...)`
- `sample_fol_problem(...)`

Behavior:
- layered transitions `src_layer -> src_layer+1`
- Horn clauses with body/head atom tuples
- function-free terms (variables/constants only)
- generalized modus ponens via unification against current ground facts

### 2) New FOL tokenizer
Add `task/layer_gen/util/tokenize_layer_fol.py`:
- API parity with current layer tokenizer:
  - `tokenize_prompt`, `encode_completion`, `tokenize_example`
  - `decode_prompt`, `decode_completion_text`
  - `to_dict`/`from_dict`
- Keep SEP/EOT and compact text format:
  - prompt example:
`Parent(alice,bob),Parent(bob,carol)⊢Grandparent(alice,carol)`
  - completion example:
`Parent(alice,bob)∧Parent(bob,carol)→Grandparent(alice,carol)<EOT>`

### 3) New dataset generator
Add `task/layer_gen/generate_layer_fol.py`:
- mirror `generate_layer.py` structure
- same metadata/stat style
- same sharding and per-distance orchestration
- same record schema (`prompt`, `completions`, `distance`, `src_layer`, etc.)

CLI knobs (FOL-specific):
- `--predicates-per-layer`
- `--arity-max`
- `--constants`
- `--vars-per-rule-max`

### 4) New runtime task loader
Add `task/layer_fol.py` with `FOLLayerTask`:
- constructor parity with `LayerTask`
- online/offline modes
- `autoregressive` and `all_at_once`
- `fixed_length_mode` support with `global_max` behavior analogous to existing
task

### 5) Completion validity/matching utilities
In `task/layer_fol.py`:
- decode completion
- check it is an instance of a rule-bank clause at `src_layer`
- check body applicability to prompt facts
- expose aggregate metrics (decode error, unknown rule, inapplicable rule,
accuracy)

### 6) Tests
Add:
- `tests/test_fol_rule_bank.py`
- `tests/test_layer_fol_tokenize.py`
- `tests/test_generate_layer_fol.py`
- `tests/test_layer_fol_task.py`

Coverage goals:
- unification correctness and variable-sharing edge cases
- tokenizer roundtrip + metadata integrity
- generator count/stat invariants
- task batching/objective/global-max parity with `LayerTask`
- validity-checker failure buckets

## Acceptance Criteria
- New FOL tests pass.
- Existing layer/propositional tests remain green.
- Tiny generated FOL dataset loads and iterates in offline mode.
- Online FOL task produces stable shapes under global fixed length.

quantifier rules.
- Quantifiers remain implicit universal over rule vars.
- Existing training loop remains unchanged because `(xs, ys)` interface is
preserved.

