#!/bin/sh
#SBATCH --account=zi
#SBATCH --job-name=sg_sweep
#SBATCH -c 1
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=3gb
#SBATCH --cpus-per-task=2

echo "$1"
echo "$2"
wandb agent --count="$2" "$1"