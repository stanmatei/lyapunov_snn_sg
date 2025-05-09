#!/bin/sh
#SBATCH --job-name=d3
#SBATCH -c 1
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=4b
#SBATCH --cpus-per-task=2

echo "$1"
echo "$2"
wandb agent --count="$2" "$1"