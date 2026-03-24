#!/bin/bash
#SBATCH -c 48
#SBATCH -t 12:00:00
#SBATCH -p sapphire
#SBATCH --mem=64G
#SBATCH -o log.hybrid_icl.%j.out
#SBATCH -e log.hybrid_icl.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ecakar@college.harvard.edu
#SBATCH --account=kempner_pehlevan_lab

# Generate hybrid ICL offline datasets (all three eval conditions).
# Pure CPU — no GPU needed. Uses 48 cores for parallel distance generation.
#
# Output: $OUT_DIR/{hybrid_icl_train,hybrid_icl_rule_gen,hybrid_icl_pred_gen}/
#
# Estimated time: ~6-10h for 50k examples/distance at 16 depths x 3 conditions.
# The demo augmentation adds overhead vs internalized.

set -euo pipefail

REPO_ROOT="/n/home00/ecakar/HRUs"
cd "${REPO_ROOT}"

SEED=${SEED:-42}
OUT_DIR=${OUT_DIR:-"/n/home00/ecakar/scratch/fol_seed${SEED}"}
MIN_D=${MIN_D:-1}
MAX_D=${MAX_D:-16}
EXAMPLES=${EXAMPLES:-50000}

# Hybrid ICL parameters
FRESH_PREDS=${FRESH_PREDS:-8}
FRESH_RULES=${FRESH_RULES:-8}
P_FRESH=${P_FRESH:-0.5}
PRED_TRAIN_FRAC=${PRED_TRAIN_FRAC:-0.5}
MAX_N_DEMOS=${MAX_N_DEMOS:-16}
DEMO_DIST=${DEMO_DIST:-"zipf_per_rule"}
DEMO_ALPHA=${DEMO_ALPHA:-1.0}

source "${REPO_ROOT}/.venv/bin/activate"

echo "=== Generating hybrid ICL conditions ==="
echo "Seed: ${SEED}"
echo "Output: ${OUT_DIR}"
echo "Depths: ${MIN_D}..${MAX_D}"
echo "Examples per depth: ${EXAMPLES}"
echo "Fresh preds/layer: ${FRESH_PREDS}, Fresh rules/transition: ${FRESH_RULES}"
echo "p_fresh: ${P_FRESH}, pred_train_frac: ${PRED_TRAIN_FRAC}"
echo "Max demos: ${MAX_N_DEMOS}, Distribution: ${DEMO_DIST}, Alpha: ${DEMO_ALPHA}"
echo "Workers: 48"
echo ""

python -m task.layer_gen.generate_fol_dataset \
    --out-dir "${OUT_DIR}" \
    --seed "${SEED}" \
    --conditions hybrid_icl_train,hybrid_icl_rule_gen,hybrid_icl_pred_gen \
    --min-distance "${MIN_D}" \
    --max-distance "${MAX_D}" \
    --examples-per-distance "${EXAMPLES}" \
    --examples-per-shard 50000 \
    --workers 48 \
    --completion-format single \
    --fresh-predicates-per-layer "${FRESH_PREDS}" \
    --fresh-rules-per-transition "${FRESH_RULES}" \
    --p-fresh "${P_FRESH}" \
    --pred-train-frac "${PRED_TRAIN_FRAC}" \
    --fresh-predicate-name-len 4 \
    --max-n-demos "${MAX_N_DEMOS}" \
    --min-n-demos 1 \
    --demo-distribution "${DEMO_DIST}" \
    --demo-alpha "${DEMO_ALPHA}" \
    --demo-ranked

echo ""
echo "=== Done ==="
for cond in hybrid_icl_train hybrid_icl_rule_gen hybrid_icl_pred_gen; do
    echo "--- ${cond} ---"
    ls -la "${OUT_DIR}/${cond}/" 2>/dev/null || echo "  (not found)"
done
