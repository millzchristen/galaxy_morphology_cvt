#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

# module load cuda/11.1.1
# source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate jades-cvt

BASELINE_CKPT="${BASELINE_CKPT:-runs/baseline_*/model_best.pth}"
# Resolve glob to a single path if needed
BASELINE_CKPT=$(ls -1 $BASELINE_CKPT 2>/dev/null | head -n 1)

python tools/finetune.py \
  --cfg configs/cvt-13-224x224.yaml \
  --pretrain "$BASELINE_CKPT" \
  --dataset_root data/galaxy_zoo \
  --work_dir runs/domain_adapt \
  --target_res 64 \
  --noise_max 0.15 \
  --noise_steps 2 \
  --res_steps 4 \
  --epochs_per_phase 50 \
  --epochs_joint 50 \
  --es_patience 5 \
  --es_min_delta 1e-4 \
  --run_name finetune2
