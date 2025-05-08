#!/bin/bash

start_sweep_job=start_sweep_job.sh
sbatch $start_sweep_job
sweep_id=$(< sweep_id.txt)
echo "$sweep_id"
jobfile=slurm_pass_forward.sh

for ((i=1; i<=$1; i++)); do
    sbatch $jobfile "$sweep_id"
    sleep 30
    echo "waiting"
done

wait
