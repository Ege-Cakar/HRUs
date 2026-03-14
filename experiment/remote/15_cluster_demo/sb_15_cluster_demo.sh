#!/bin/bash
#SBATCH -c 8
#SBATCH -t 1-00:00
#SBATCH -p kempner_h100,kempner
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -o log.%A.%a.out
#SBATCH -e log.%A.%a.err
#SBATCH --array=1-96%12
#SBATCH --mail-type=END
#SBATCH --mail-user=wtong@g.harvard.edu
#SBATCH --account=kempner_grads

source ../../../.venv/bin/activate
python run.py ${SLURM_ARRAY_TASK_ID}
