#!/bin/bash

wandb sweep --update $1 sweep.yaml

for ((i=1; i<=$2; i++)); do
    jobfile="sgf_job"
    
    echo "#!/bin/sh" >> $jobfile
    echo "#SBATCH --job-name=d3" >> $jobfile
    echo "#SBATCH -c 1" >> $jobfile
    echo "#SBATCH --time=8:00:00" >> $jobfile
    echo "#SBATCH --mem-per-cpu=2gb" >> $jobfile
    echo "#SBATCH --cpus-per-task=1" >>$jobfile
    echo "wandb agent $1" >>$jobfile
    echo "date" >> $jobfile
    
    sbatch $jobfile
    > $jobfile
    echo "waiting"
done
wait
