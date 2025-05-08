#!/bin/sh
#SBATCH --job-name=d3
#SBATCH -c 1
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=2gb
#SBATCH --cpus-per-task=1
echo $1
wandb agent $1
