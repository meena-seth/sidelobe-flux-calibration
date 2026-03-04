#!/bin/bash
#SBATCH --account=def-halpern
#SBATCH --export=NONE
#SBATCH --time=07:00:00
#SBATCH --mem-per-cpu=4096M
#SBATCH --cpus-per-task=1
#SBATCH --job-name=flux_cal
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load python/3.11.5
module load mpi4py/4.0.3
module load gcc opencv/4.12.0
module load gcc arrow/14.0.0
 
source ~/venvs/holo/bin/activate

python_dir=/home/mseth2/scratch/sidelobe-flux-calibration/combining_fluxcal_results.py

python "$python_dir"
