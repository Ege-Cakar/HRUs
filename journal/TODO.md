# TODO

## Architecture
- [ ] **RecursiveArchitecture class**: Separate class for recursive models (prelude/core/coda).
  - Core block weights are shared (weight-tied) across loop iterations
  - `n_recurrences` parameter: can differ between train and test time
  - Input injection at each loop iteration (concat + linear projection)
  - For hybrid recursive, core block size must be multiple of 4
  - Variants: RecursiveTransformer (core=4×Attn), RecursiveHybrid (core=GDN,GDN,GDN,Attn)
- [ ] **Chunkwise parallel GDN**: Current GDN uses sequential scan. For longer sequences,
  implement chunkwise parallel form for better GPU utilization. Functionally identical,
  purely a performance optimization.
- [ ] **Adaptive depth / halting**: Implement entropy-regularized gating (Ouro/Zhu et al. 2025)
  and oracle exiting (Bae et al. 2024) for the recursive architectures.
- [ ] **Pretrained retrofit**: Model surgery to convert Qwen 3 / Qwen 3.5 into recursive
  variants following McLeish et al. 2025. Layer selection, weight tying, curriculum of
  recurrences during continued pretraining.

## Tasks
- [ ] **Hybrid ICL integration testing**: Full end-to-end test of hybrid ICL condition
  (internalized + novel in-context rules) with training loop.
- [ ] **ICL task (GD connection)**: Clean task designed to isolate in-context learning
  capacity, connecting to theoretical results on looped transformers implementing
  gradient descent (von Oswald et al. 2023, Gatmiry et al. 2024).
- [ ] **Depth curriculum**: Add curriculum support for training depth D — start with small D,
  gradually increase. Related to recursive recurrence curriculum but applies to all conditions.
- [ ] **Math task setup**: Fine-tuning pipeline for Qwen 3.5 on math reasoning benchmarks
  via train_hf.py / Accelerate. Evaluation harness for MATH/GSM8K or similar.

## Experiments
- [ ] **SLURM scripts for offline FOL dataset generation**: CPU-heavy jobs on sapphire partition.
- [ ] **Scaling sweep configs**: 10 synthetic model sizes (~10M-600M params) across all 4
  architecture conditions, both FOL conditions.
- [ ] **Pretrained experiment configs**: Qwen 3.5 at 0.8B, 2B, 4B, 9B, 27B — standard vs
  recursive retrofit, fine-tuned on math task.

## Ablations (Section 5 of plan)
- [ ] **Attention ratio in recurrent core**: Compare 3:1, 1:1, and higher GDN ratios
  specifically within the looped core.
- [ ] **Recurrences vs depth D**: Vary test-time T and derivation depth D independently.
  Characterize compute-accuracy tradeoff and extrapolation.
- [ ] **Halting strategies**: Fixed T vs adaptive depth (Ouro) vs oracle exit (Bae et al.).

## Infrastructure
- [ ] **Offline dataset loading**: Verify grain-based ArrayRecord loading path works with
  new generate_fol_dataset.py shards for all conditions.
- [ ] **W&B experiment tracking**: Ensure wandb_utils integration works with ArchConfig
  (currently only tested with TransformerConfig).
