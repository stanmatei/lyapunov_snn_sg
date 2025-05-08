#!/bin/bash

sweep_id=$(wandb sweep --entity=snn_nlp --project=lyapunov_snn sweep.yaml)
echo $sweep_id
sweep_out=${}
jobfile=slurm_pass_forward.sh

for ((i=1; i<=$1; i++)); do
    sbatch $jobfile $sweep_id
    sleep 30
    echo "waiting"
done
wait
