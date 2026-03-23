#!/bin/bash
#SBATCH -c 48
#SBATCH -t 16:00:00
#SBATCH -p sapphire
#SBATCH --mem=64G
#SBATCH -o log.generate_all.%j.out
#SBATCH -e log.generate_all.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ecakar@college.harvard.edu
#SBATCH --account=kempner_pehlevan_lab

# Generate ALL offline datasets in a single job:
#   - internalized (no demos)
#   - hybrid_icl_train (fresh rules from training pool)
#   - hybrid_icl_rule_gen (same preds, new rules)
#   - hybrid_icl_pred_gen (new preds + new rules)
#
# All conditions share the same base rule bank (deterministic from seed).
# Pure CPU — no GPU needed.
#
# Estimated time: ~10-16h for 50k examples/depth x 16 depths x 4 conditions.

set -euo pipefail

SEED=${SEED:-42}
OUT_DIR=${OUT_DIR:-"data/fol_seed${SEED}"}
MIN_D=${MIN_D:-1}
MAX_D=${MAX_D:-16}
EXAMPLES=${EXAMPLES:-50000}

source ../../../.venv/bin/activate

echo "============================================================"
echo "  FOL Dataset Generation — All Conditions"
echo "============================================================"
echo "Seed:              ${SEED}"
echo "Output:            ${OUT_DIR}"
echo "Depths:            ${MIN_D}..${MAX_D}"
echo "Examples/depth:    ${EXAMPLES}"
echo "CPUs:              48"
echo "============================================================"
echo ""

python -m task.layer_gen.generate_fol_dataset \
    --out-dir "${OUT_DIR}" \
    --seed "${SEED}" \
    --conditions internalized,hybrid_icl_train,hybrid_icl_rule_gen,hybrid_icl_pred_gen \
    --min-distance "${MIN_D}" \
    --max-distance "${MAX_D}" \
    --examples-per-distance "${EXAMPLES}" \
    --examples-per-shard 50000 \
    --workers 48 \
    --completion-format single \
    --n-layers 16 \
    --predicates-per-layer 8 \
    --rules-per-transition 32 \
    --arity-min 1 \
    --arity-max 3 \
    --vars-per-rule-max 4 \
    --constants "a,b,c,d" \
    --initial-ant-max 3 \
    --fresh-predicates-per-layer 8 \
    --fresh-rules-per-transition 8 \
    --p-fresh 0.5 \
    --pred-train-frac 0.5 \
    --fresh-predicate-name-len 4 \
    --max-n-demos 16 \
    --min-n-demos 1 \
    --demo-distribution zipf_per_rule \
    --demo-alpha 1.0 \
    --demo-ranked

echo ""
echo "============================================================"
echo "  Generation complete"
echo "============================================================"
echo ""
echo "Output structure:"
ls -la "${OUT_DIR}/"
echo ""
for cond in internalized hybrid_icl_train hybrid_icl_rule_gen hybrid_icl_pred_gen; do
    n_shards=$(find "${OUT_DIR}/${cond}" -name "*.array_record" 2>/dev/null | wc -l)
    echo "  ${cond}: ${n_shards} shards"
done
