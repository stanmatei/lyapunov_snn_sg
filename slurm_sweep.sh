#!/bin/bash

wandb sweep --update $1 sweep.yaml
jobfile=slurm_pass_forward.sh

for ((i=1; i<=$2; i++)); do
    sbatch $jobfile $1
    sleep 30
    echo "waiting"
done
wait
