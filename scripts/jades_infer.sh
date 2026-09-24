#!/bin/bash
#SBATCH --job-name=jades_infer
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/jades_infer_%j.out
#SBATCH --error=logs/jades_infer_%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

# module load cuda/11.1.1
# source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate jades-cvt

CHECKPOINT="${CHECKPOINT:-runs/fixmatch/fixtest/tau_0_90/model_epoch3.pth}"
MASKED_DIR="${MASKED_DIR:-data/jades/test}"
ORIGINAL_DIR="${ORIGINAL_DIR:-data/jades/cutouts}"
METADATA_FILE="${METADATA_FILE:-data/jades/cutout_metadata.json}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/jades_predictions/tau_0_75}"
FEATURES_OUTPUT="${FEATURES_OUTPUT:-runs/jades_features.npz}"
TAU="${TAU:-0.75}"

python tools/jades_infer.py \
  --cfg configs/cvt-13-224x224.yaml \
  --checkpoint "$CHECKPOINT" \
  --jades_dir "$MASKED_DIR" \
  --original_dir "$ORIGINAL_DIR" \
  --metadata_file "$METADATA_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --features_output "$FEATURES_OUTPUT" \
  --tau "$TAU" \
  --batch_size 32 \
  --num_workers 8 \
  --target_res 64
