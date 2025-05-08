#!/bin/bash

for i in {1..2}; do 
    jobfile="sgf_job"
    
    echo "#!/bin/sh" >> $jobfile
    echo "#SBATCH --job-name=d3" >> $jobfile
    echo "#SBATCH -c 1" >> $jobfile
    echo "#SBATCH --time=24:00:00" >> $jobfile
    echo "#SBATCH --mem-per-cpu=2gb" >> $jobfile
    echo "#SBATCH --cpus-per-task=1" >>$jobfile
    echo "wandb agent snn_nlp/lyapunov_snn/bj0q23cr" >>$jobfile
    echo "date" >> $jobfile
    
    sbatch $jobfile
    echo "waiting"
done
wait
