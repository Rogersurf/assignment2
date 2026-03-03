#!/bin/bash

#SBATCH --job-name=green_patent
#SBATCH --output=logs/train_%j.out
#SBATCH --partition=l4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00

NUM_GPUS=4

echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

source .venv/bin/activate

# GPU setup
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Para usar as 4 GPUs corretamente
torchrun \
    --nproc_per_node=$NUM_GPUS \
    assignment02_m4.py

echo "[DONE] Training complete."