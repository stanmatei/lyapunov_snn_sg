#!/bin/bash

#SWEEP_ID="snn_nlp/lyapunov_snn/m4poin1v"

#wandb sweep --update $SWEEP_ID sweep.yaml

for ((i=1; i<=$2; i++)); do
    jobfile="sgf_job"
    
    echo "#!/bin/sh" >> $jobfile
    echo "#SBATCH --job-name=d3" >> $jobfile
    echo "#SBATCH -c 1" >> $jobfile
    echo "#SBATCH --time=24:00:00" >> $jobfile
    echo "#SBATCH --mem-per-cpu=2gb" >> $jobfile
    echo "#SBATCH --cpus-per-task=1" >>$jobfile
    echo "wandb agent snn_nlp/lyapunov_snn/m4poin1v" >>$jobfile
    echo "date" >> $jobfile
    
    sbatch $jobfile
    echo "waiting"
done
wait
