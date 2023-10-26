#!/bin/bash
#SBATCH --chdir=/home/ktanner/TPIV---Logistic-Classification/experiments
#SBATCH --job-name=TPIV-Adversarial
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --output=out.txt
#SBATCH --error=error.txt
#SBATCH --time=36:00:00

module purge
module load gcc openmpi python/3.10.4
source /home/ktanner/venvs/adv/bin/activate

srun --mpi=pmi2 python3 ./sweep.py sweep_experiment.json

deactivate