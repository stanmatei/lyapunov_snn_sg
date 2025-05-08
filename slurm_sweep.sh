#!/bin/bash
sweep_id = $(python start_sweep.py --entity=snn_nlp --project=lyapunov_snn --config=sweep.yaml)
echo "$sweep_id"
jobfile=slurm_pass_forward.sh

for ((i=1; i<=$1; i++)); do
    sbatch $jobfile $sweep_id
    sleep 30
    echo "waiting"
done
wait
