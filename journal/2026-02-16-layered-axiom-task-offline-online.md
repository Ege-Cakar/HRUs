# 2026-02-16 - Layered Axiom Task (Offline + Online)

## Task Summary
- Implemented a new layered propositional reasoning task where propositions are indexed by `(layer, id)` and inference uses a fixed layer-specific axiom bank.
- Inputs are sequents from layer `i` to a single goal proposition in layer `i+n`; targets are next-axiom statements.
- Added support for both objectives:
  - `autoreg`: next-axiom completion with prompt masking.
  - `first_step`: all-at-once first-action target (Mixer-compatible).

## Implementation Strategy
- Reused existing project patterns (ArrayRecord + Grain loaders, metadata stats, tokenized prompt/completion format, pytest structure).
- Built a shared rule-bank/sampling core so offline generation and online mode use identical semantics.
- Made offline pre-generated datasets the default and exposed online fresh sampling via a mode flag in one task API.
- Persisted `rule_bank.json` with generated datasets to enable exact validity checks of model outputs.
- Kept transitions monotonic (union closure) and used single sampled proof paths for deterministic supervision per sampled example.

## Components Added
- `task/layer_axiom.py`: unified task API (`mode=offline|online`, `objective=autoreg|first_step`) + validity helper.
- `task/layer_gen/generate_layer_axiom.py`: offline dataset generator with distance buckets.
- `task/layer_gen/util/rule_bank.py`: fixed layer-specific axiom bank generation/serialization.
- `task/layer_gen/util/tokenize_layer_axiom.py`: encode/decode for prompts and axiom statements.
- New tests for tokenization, task loaders, generator metadata/shards, and validity lookup.

## Naming Update
- Renamed interface terms from `k`/`kprime` to `k_in`/`k_out` (`k_in_max`, `k_out_max`) across CLI, config, code, and tests.
