#!/bin/bash
#SBATCH --job-name=fixmatch
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/fixmatch_%j.out
#SBATCH --error=logs/fixmatch_%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

# module load cuda/11.1.1
# source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate jades-cvt

ADAPT_CKPT="${ADAPT_CKPT:-runs/domain_adapt/finetune2_*/model_last.pth}"
ADAPT_CKPT=$(ls -1 $ADAPT_CKPT 2>/dev/null | head -n 1)

python tools/fixmatch.py \
  --cfg configs/cvt-13-224x224.yaml \
  --pretrain "$ADAPT_CKPT" \
  --run_name fixtest \
  --dataset_root data/imagenet \
  --work_dir runs/fixmatch \
  --tau_list 0.85,0.90 \
  --epochs 10 \
  --batch_size 8 \
  --lr 1e-5 \
  --mu 3 \
  --lambda_u 1.0 \
  --target_res 64 \
  --noise_std 0.0 \
  --num_workers 8 \
  --seed 42 \
  --num_ops 2 \
  --magnitude 4 \
  --blue_shrink 0.05 \
  --include_blue_in_pool
