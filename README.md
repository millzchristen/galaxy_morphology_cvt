# galaxy_morphology_cvt

Minimal reproduction code for galaxy morphology classification with a
[CvT-13](https://arxiv.org/abs/2103.15808) backbone: fine-tune on Galaxy Zoo 2,
adapt to low-resolution noisy imaging, refine with FixMatch, and run inference
on JADES cutouts.

This repository contains **code and configs only**. Datasets and trained
checkpoints are not included (see below).

## Classes

| Folder   | Morphology   |
|----------|--------------|
| class1   | Round        |
| class2   | In-between   |
| class3   | Cigar        |
| class4   | Edge-on      |
| class5   | Spiral       |

## Setup

```bash
# Option A: conda (exported HPC environment; name: jades-cvt)
conda env create -f environment.yml
conda activate jades-cvt

# Option B: pip (plus a matching PyTorch + CUDA install)
pip install -r requirements.txt
pip install torch torchvision  # install the build matching your CUDA
```

Download ImageNet-pretrained CvT-13 weights into `pretrain/`:

```
pretrain/CvT-13-224x224-IN-1k.pth
```

Upstream weights: [Microsoft CvT model zoo](https://github.com/microsoft/CvT).

## Data layout (not in git)

```
data/
  galaxy_zoo/
    train/class{1-5}/
    val/class{1-5}/
  imagenet/                 # ImageFolder layout used by FixMatch / test.py
    train/class{1-5}/
    val/class{1-5}/
    test/class{1-5}/
    test_corrupt/class{1-5}/
  jades/
    test/                   # high-SNR masked cutouts for inference
    cutouts/                # optional originals for viewer copy-out
    cutout_metadata.json
pretrain/
  CvT-13-224x224-IN-1k.pth
runs/                       # created by training (gitignored)
```

## Reproduce the pipeline

Run all commands from the **repository root**.

### 1. Baseline fine-tune (Galaxy Zoo)

```bash
python tools/train.py \
  --cfg configs/cvt-13-224x224.yaml \
  --pretrain pretrain/CvT-13-224x224-IN-1k.pth \
  --dataset_root data/galaxy_zoo \
  --work_dir runs \
  --epochs 15 \
  --run_name baseline
```

### 2. Domain adaptation (resolution + noise curriculum)

```bash
python tools/finetune.py \
  --cfg configs/cvt-13-224x224.yaml \
  --pretrain runs/baseline_<host>/model_best.pth \
  --dataset_root data/galaxy_zoo \
  --work_dir runs/domain_adapt \
  --target_res 64 \
  --noise_max 0.15 \
  --noise_steps 2 \
  --res_steps 4 \
  --epochs_per_phase 50 \
  --epochs_joint 50 \
  --run_name finetune2
```

### 3. FixMatch semi-supervised refinement

```bash
python tools/fixmatch.py \
  --cfg configs/cvt-13-224x224.yaml \
  --pretrain runs/domain_adapt/finetune2_<host>/model_last.pth \
  --dataset_root data/imagenet \
  --work_dir runs/fixmatch \
  --run_name fixtest \
  --tau_list 0.85,0.90 \
  --epochs 10 \
  --target_res 64 \
  --blue_shrink 0.05 \
  --include_blue_in_pool
```

### 4. Evaluate on corrupted GZ2 test set

```bash
python tools/test.py \
  --cfg configs/cvt-13-224x224.yaml \
  DATASET.ROOT data/imagenet/ \
  DATASET.TEST_SET test_corrupt \
  TEST.MODEL_FILE runs/domain_adapt/finetune2_<host>/model_last.pth
```

### 5. JADES inference

```bash
python tools/jades_infer.py \
  --cfg configs/cvt-13-224x224.yaml \
  --checkpoint runs/fixmatch/fixtest/tau_0_90/model_epoch3.pth \
  --jades_dir data/jades/test \
  --original_dir data/jades/cutouts \
  --metadata_file data/jades/cutout_metadata.json \
  --output_dir runs/jades_predictions/tau_0_75 \
  --features_output runs/jades_features.npz \
  --tau 0.75 \
  --target_res 64
```

Optional SLURM wrappers with the same relative paths live in `scripts/`.

## Checkpoints

Trained weights from this work are **not** stored in this repository. Release
archives (baseline / adapted / FixMatch) will be linked here when published
(Zenodo / Hugging Face / institutional storage).

## Citation

- CvT architecture: Wu et al., *CvT: Introducing Convolutions to Vision Transformers*, ICCV 2021.
- Upstream code: [microsoft/CvT](https://github.com/microsoft/CvT).
- This morphology adaptation pipeline: cite the associated paper / thesis when available.

## Layout

```
configs/          # CvT-13 YAML
lib/              # model, data, optim, utils
tools/            # train, finetune, fixmatch, test, jades_infer
scripts/          # optional SLURM/bash wrappers
```
