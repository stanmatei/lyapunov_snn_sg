#!/bin/bash
file_name="test.txt"
wandb sweep --entity=snn_nlp --project=lyapunov_snn sweep.yaml
jobfile=slurm_pass_forward.sh

for ((i=1; i<=$1; i++)); do
    sbatch $jobfile $sweep_id
    sleep 30
    echo "waiting"
done
wait
