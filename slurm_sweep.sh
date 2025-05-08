#!/bin/bash

sweep_id=$(wandb sweep --entity=snn_nlp --project=lyapunov_snn sweep.yaml | awk '/ID:/{print $2}')
echo $sweep_id
jobfile=slurm_pass_forward.sh

for ((i=1; i<=$1; i++)); do
    sbatch $jobfile $sweep_id
    sleep 30
    echo "waiting"
done
wait
