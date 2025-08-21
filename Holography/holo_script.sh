#!/bin/bash
#SBATCH --account=def-halpern
#SBATCH --ntasks=1               # number of MPI processes
#SBATCH --mem=8GB      # memory; default unit is megabytes
#SBATCH --time=0-01:00           # time (DD-HH:MM)
module use /project/rpp-chime/chime/chime_env/modules/modulefiles/
module load chime/python/2025.03
source ~/meenaseth/bin/activate
python /home/mseth2/scratch/nrao/trying_on_cedar/holo_script.py $1
