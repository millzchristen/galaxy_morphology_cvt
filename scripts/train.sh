#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

# module load cuda/11.1.1   # uncomment on clusters that require it
# source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate jades-cvt

python tools/train.py \
  --cfg configs/cvt-13-224x224.yaml \
  --pretrain pretrain/CvT-13-224x224-IN-1k.pth \
  --dataset_root data/galaxy_zoo \
  --work_dir runs \
  --epochs 15 \
  --run_name baseline
