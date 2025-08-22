#!/bin/sh
for i in $(seq 0 1023);
do
    sbatch holo_script.sh $i
done

