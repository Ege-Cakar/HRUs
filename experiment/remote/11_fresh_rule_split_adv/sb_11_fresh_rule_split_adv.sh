#!/bin/bash
#SBATCH -c 8
#SBATCH -t 2-00:00
#SBATCH -p kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -o log.%A.%a.out
#SBATCH -e log.%A.%a.err
#SBATCH --array=1-4
#SBATCH --mail-type=END
#SBATCH --mail-user=wtong@g.harvard.edu
#SBATCH --account=kempner_pehlevan_lab

source ../../../.venv/bin/activate
python run.py ${SLURM_ARRAY_TASK_ID}
