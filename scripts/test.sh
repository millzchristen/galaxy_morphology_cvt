#!/bin/bash
#SBATCH --job-name=test_corrupt
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

# module load cuda/11.1.1
# source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate jades-cvt

MODEL_FILE="${MODEL_FILE:-runs/domain_adapt/finetune2_*/model_last.pth}"
MODEL_FILE=$(ls -1 $MODEL_FILE 2>/dev/null | head -n 1)

python tools/test.py \
  --cfg configs/cvt-13-224x224.yaml \
  --run_name test_corrupt \
  DATASET.ROOT data/imagenet/ \
  DATASET.TEST_SET test_corrupt \
  TEST.MODEL_FILE "$MODEL_FILE"
