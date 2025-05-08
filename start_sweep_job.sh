#!/bin/sh
#SBATCH --job-name=d3
#SBATCH -c 1
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=2gb
#SBATCH --cpus-per-task=1
python start_sweep.py --entity=snn_nlp --project=lyapunov_snn --config=sweep.yaml