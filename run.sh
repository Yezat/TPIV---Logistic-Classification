#!/bin/bash
#SBATCH --chdir=/home/ktanner/TPIV---Logistic-Classification
#SBATCH --job-name=TPIV-Adversarial
#SBATCH --nodes=1
#SBATCH --ntasks=50
#SBATCH --cpus-per-task=1
#SBATCH --mem=400G
#SBATCH --output=out.txt
#SBATCH --error=error.txt
#SBATCH --time=36:00:00

module purge
module load gcc openmpi python/3.10.4
source /home/ktanner/venvs/adv/bin/activate

srun --mpi=pmi2 python3 ./sweep.py

deactivate