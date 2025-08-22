#!/bin/bash
#SBATCH --account=def-istairs
#SBATCH --ntasks=1               # number of MPI processes
#SBATCH --mem=8GB      # memory; default unit is megabytes
#SBATCH --time=0-01:00           # time (DD-HH:MM)
module use /project/rpp-chime/chime/chime_env/modules/modulefiles/
module load chime/python/2025.03
source ~/holography/bin/activate
python holo_script.py $1
