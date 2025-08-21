#!/bin/sh
for i in $(seq 0 1023);
do
    sbatch /home/mseth2/scratch/nrao/trying_on_cedar/holo_script.sh $i
done

