#!/bin/bash
#SBATCH -c 48
#SBATCH -t 8:00:00
#SBATCH -p sapphire
#SBATCH --mem=64G
#SBATCH -o log.internalized.%j.out
#SBATCH -e log.internalized.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ecakar@college.harvard.edu
#SBATCH --account=kempner_pehlevan_lab

# Generate internalized-rules offline dataset.
# Pure CPU — no GPU needed. Uses 48 cores for parallel distance generation.
#
# Output: $OUT_DIR/internalized/distance_001/ ... distance_016/
#
# Estimated time: ~2-4h for 50k examples/distance at 16 depths.

set -euo pipefail

REPO_ROOT="/n/home00/ecakar/HRUs"
cd "${REPO_ROOT}"

SEED=${SEED:-42}
OUT_DIR=${OUT_DIR:-"/n/home00/ecakar/scratch/fol_seed${SEED}"}
MIN_D=${MIN_D:-1}
MAX_D=${MAX_D:-16}
EXAMPLES=${EXAMPLES:-50000}

source "${REPO_ROOT}/.venv/bin/activate"

echo "=== Generating internalized condition ==="
echo "Seed: ${SEED}"
echo "Output: ${OUT_DIR}"
echo "Depths: ${MIN_D}..${MAX_D}"
echo "Examples per depth: ${EXAMPLES}"
echo "Workers: 48"
echo ""

python -m task.layer_gen.generate_fol_dataset \
    --out-dir "${OUT_DIR}" \
    --seed "${SEED}" \
    --conditions internalized \
    --min-distance "${MIN_D}" \
    --max-distance "${MAX_D}" \
    --examples-per-distance "${EXAMPLES}" \
    --examples-per-shard 50000 \
    --workers 48 \
    --completion-format single

echo ""
echo "=== Done ==="
echo "Output: ${OUT_DIR}/internalized/"
ls -la "${OUT_DIR}/internalized/"
