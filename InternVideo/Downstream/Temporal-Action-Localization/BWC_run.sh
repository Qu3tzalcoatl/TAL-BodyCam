#!/bin/bash
#SBATCH --job-name=BWC
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1
#SBATCH --account=def-panos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=out/log-%x-%j.out
#SBATCH --mem-per-cpu=32G

module load  StdEnv/2020  cuda cudnn
module load gcc opencv

nvidia-smi

source  ../../../ENV/bin/activate

echo "Testing..."

python -u ./train_eval.py ./configs/BWC.yaml --output BWC_out 2>&1 | tee BWC_out.log


