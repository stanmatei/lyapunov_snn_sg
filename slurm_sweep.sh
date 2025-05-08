#!/bin/bash

python start_sweep.py --entity=snn_nlp --project=lyapunov_snn --config=sweep.yaml
sweep_id_file=sweep_id.txt
sweep_id=$(< "$sweep_id_file")
echo "$sweep_id"
jobfile=slurm_pass_forward.sh

for ((i=1; i<=$1; i++)); do
    sbatch $jobfile "$sweep_id" "$2"
    sleep 2
    echo "waiting"
done

wait
