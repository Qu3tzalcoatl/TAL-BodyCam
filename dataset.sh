#!/bin/bash
#SBATCH --job-name=BWC_dataset
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1
#SBATCH --account=def-panos
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --output=out/log-%x-%j.out
#SBATCH --mem-per-cpu=16G

module load  StdEnv/2020

nvidia-smi

source ENV/bin/activate

echo "Testing..."

python mp4_to_npy.py


