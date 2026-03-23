# Dataset Generation SLURM Scripts

Generate offline FOL datasets on the Kempner cluster. All scripts use CPU-only
nodes (`sapphire` partition) — no GPU needed.

## Quick start

```bash
# Generate everything in one job (~10-16h):
sbatch sb_generate_all.sh

# Or generate conditions separately:
sbatch sb_generate_internalized.sh   # ~2-4h
sbatch sb_generate_hybrid_icl.sh     # ~6-10h
```

## Customization via environment variables

```bash
# Different seed:
SEED=123 sbatch sb_generate_all.sh

# Custom output path:
OUT_DIR=/n/holyscratch01/pehlevan_lab/ecakar/fol_data SEED=42 sbatch sb_generate_all.sh

# Smaller test run:
EXAMPLES=1000 MAX_D=4 sbatch sb_generate_all.sh

# Adjust hybrid ICL parameters:
P_FRESH=0.3 MAX_N_DEMOS=8 sbatch sb_generate_hybrid_icl.sh
```

## Output structure

```
data/fol_seed42/
  rule_bank.json                    # shared base bank
  hybrid_icl_bank.json              # hybrid ICL bank (if generated)
  internalized/
    distance_001/ ... distance_016/
    metadata.json
  hybrid_icl_train/
    distance_001/ ... distance_016/
    metadata.json
  hybrid_icl_rule_gen/
    distance_001/ ... distance_016/
    metadata.json
  hybrid_icl_pred_gen/
    distance_001/ ... distance_016/
    metadata.json
```

## Resource estimates (50k examples/depth, 16 depths)

| Condition | Time | Memory | Notes |
|-----------|------|--------|-------|
| internalized | 2-4h | ~32G | No demo augmentation |
| hybrid_icl (x3) | 6-10h | ~48G | Demo augmentation adds overhead |
| All combined | 10-16h | ~64G | Sequential within one job |
