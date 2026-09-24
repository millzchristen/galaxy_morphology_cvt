"""
Progressive domain-shift adaptation CvT-13 galaxy morphology classifier.

Goal: make the model robust to (1) lower resolution and (2) higher noise without
      destroying the fine-tuned galaxy weights.

Strategy
--------
  Phase 1 – Resolution curriculum  (clean images, res stepped 224→target_res via 2^x-aligned steps)
  Phase 2 – Noise curriculum  (target res fixed, noise σ ramped 0→noise_max)
  Phase 3 – Joint fine-tune  (target res + noise_max, very small LR)

  Rationale: resolution is the harder perturbation for CvT (destroys patch structure);
  doing it first means the noise curriculum operates in the actual target spatial regime.
  Resolution steps are constrained to values that divide cleanly through the three
  stride-4/2/2 patch embedding stages, avoiding fractional token grids.

Each phase uses:
  • Differential learning rates  – early stages frozen or ~10× lower LR
  • Label smoothing + mixup      – preserve generalisation
  • OneCycleLR per phase          – stable convergence
  • EMA shadow weights           – best checkpoint is the EMA model
  
"""

from __future__ import annotations

import argparse, datetime, json, math, os, random, socket, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as tfs

# ── project imports: identical to train.py ─────────────────────────────────
import _init_paths
from lib.config import config, update_config, save_config
from lib.models import build_model
from lib.dataset.galaxy_zoo import GalaxyZoo
from lib.utils.comm import comm
from lib.utils.utils import create_logger, init_distributed, setup_cudnn


# ═══════════════════════════════════════════════════════════════════════════
#  Augmentation helpers
# ═══════════════════════════════════════════════════════════════════════════

class AddGaussianNoise:
    """Additive Gaussian noise on a normalised tensor.
    Signature matches train.py: AddGaussianNoise(mean, std).
    """
    def __init__(self, mean: float = 0., std: float = 0.0):
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return x
        # use torch.randn(tensor.size()) to match train.py exactly
        return x + torch.randn(x.size()) * self.std + self.mean

    def __repr__(self):
        return f"AddGaussianNoise(mean={self.mean}, std={self.std:.4f})"


def build_transform(img_size: int, noise_std: float, augment: bool = True):
    """
    Build a torchvision transform pipeline.

    For training (augment=True) we add:
      • RandomHorizontalFlip + RandomVerticalFlip  (galaxies have no preferred orientation)
      • ColorJitter                                (photometric variability)
      • RandomRotation(180)                        (rotational symmetry)
      • RandomResizedCrop                          (scale / aspect jitter)
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if augment:
        pipeline = [
            tfs.Resize([img_size, img_size]),
            tfs.RandomHorizontalFlip(),
            tfs.RandomVerticalFlip(),
            tfs.RandomRotation(180),
            tfs.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            tfs.ToTensor(),
            tfs.Normalize(mean=mean, std=std),
            AddGaussianNoise(0., noise_std),
        ]
    else:
        pipeline = [
            tfs.Resize([img_size, img_size]),
            tfs.ToTensor(),
            tfs.Normalize(mean=mean, std=std),
            AddGaussianNoise(0., noise_std),  # keep noise at eval so val metric reflects target domain
        ]
    return tfs.Compose(pipeline)


# ═══════════════════════════════════════════════════════════════════════════
#  EMA (Exponential Moving Average) helper
# ═══════════════════════════════════════════════════════════════════════════

class ModelEMA:
    """Maintains a shadow copy of model weights averaged over training steps."""
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.float()

    def apply_to(self, model: nn.Module):
        model.load_state_dict({k: v.to(next(model.parameters()).dtype)
                               for k, v in self.shadow.items()})


# ═══════════════════════════════════════════════════════════════════════════
#  Early Stopping
# ═══════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Stops a phase early if val accuracy does not improve for `patience` epochs.
    reset() is called at the start of each phase so every phase gets a fresh counter.

    Args:
        patience  : epochs to wait after last improvement before stopping
        min_delta : minimum absolute improvement to count as a new best
    """
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.reset()

    def reset(self):
        self.best      = -float('inf')
        self.counter   = 0
        self.triggered = False

    def step(self, val_acc: float) -> bool:
        """Call after each epoch. Returns True if training should stop."""
        if val_acc > self.best + self.min_delta:
            self.best    = val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ═══════════════════════════════════════════════════════════════════════════
#  Mixup
# ═══════════════════════════════════════════════════════════════════════════

def mixup_batch(images, labels, alpha=0.4, num_classes=5, device='cpu'):
    """Returns mixed images and soft labels."""
    lam = np.random.beta(alpha, alpha)
    B = images.size(0)
    idx = torch.randperm(B, device=device)
    mixed = lam * images + (1 - lam) * images[idx]
    # one-hot soft labels
    y_a = torch.zeros(B, num_classes, device=device).scatter_(1, labels.unsqueeze(1), 1)
    y_b = torch.zeros(B, num_classes, device=device).scatter_(1, labels[idx].unsqueeze(1), 1)
    soft_labels = lam * y_a + (1 - lam) * y_b
    return mixed, soft_labels


class SoftCrossEntropyLoss(nn.Module):
    """Cross-entropy that accepts soft (mixed) labels."""
    def forward(self, logits, soft_labels):
        log_probs = torch.log_softmax(logits, dim=-1)
        return -(soft_labels * log_probs).sum(dim=-1).mean()


# ═══════════════════════════════════════════════════════════════════════════
#  Differential LR parameter groups
# ═══════════════════════════════════════════════════════════════════════════

def param_groups(model: nn.Module, base_lr: float, head_lr_scale: float = 10.0,
                 early_stage_scale: float = 0.1, wd: float = 0.01):
    """
    Three groups:
      • classifier head          → base_lr × head_lr_scale
      • stage2 (deepest stage)   → base_lr
      • stage0, stage1 (early)   → base_lr × early_stage_scale
    """
    head_params, stage2_params, early_params = [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('head'):
            head_params.append(param)
        elif name.startswith('stage2') or name.startswith('norm'):
            stage2_params.append(param)
        else:
            early_params.append(param)

    return [
        {'params': head_params,   'lr': base_lr * head_lr_scale,   'weight_decay': wd},
        {'params': stage2_params, 'lr': base_lr,                   'weight_decay': wd},
        {'params': early_params,  'lr': base_lr * early_stage_scale,'weight_decay': wd},
    ]


# ═══════════════════════════════════════════════════════════════════════════
#  Single epoch of training
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, scheduler, criterion,
                    device, accum_steps, num_classes, mixup_alpha,
                    print_freq=50, ema=None):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # mixup
        if mixup_alpha > 0:
            images, soft_labels = mixup_batch(images, labels, mixup_alpha,
                                              num_classes, device)
        else:
            soft_labels = torch.zeros(labels.size(0), num_classes, device=device)
            soft_labels.scatter_(1, labels.unsqueeze(1), 1)

        logits = model(images)
        loss = criterion(logits, soft_labels) / accum_steps
        loss.backward()
        running_loss += loss.item() * accum_steps

        if step % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)

        with torch.no_grad():
            pred = logits.argmax(1)
            hard_labels = labels if mixup_alpha <= 0 else soft_labels.argmax(1)
            correct += (pred == hard_labels).sum().item()
            total += labels.size(0)

        if step % (print_freq * accum_steps) == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  step {step}/{len(loader)} | loss {running_loss/step:.4f} "
                  f"| acc {correct/total:.4f} | lr {lr_now:.2e}")

    # handle leftover gradient
    if (len(loader) % accum_steps) != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if ema is not None:
            ema.update(model)

    return running_loss / len(loader), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    all_labels, all_probs = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, -1)
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return correct / max(total, 1), np.array(all_labels), np.array(all_probs)


# ═══════════════════════════════════════════════════════════════════════════
#  One full phase
# ═══════════════════════════════════════════════════════════════════════════

def run_phase(phase_name, model, ema, dataset_root, epochs, img_size, noise_std,
              base_lr, wd, batch_size, num_classes, accum_steps, mixup_alpha,
              warmup_frac, subset, seed, device, work_dir, best_val_acc,
              metrics_history, early_stopper=None,
              head_lr_scale=10.0, early_stage_scale=0.1):
    print(f"\n{'='*60}")
    print(f"  PHASE: {phase_name}  |  res={img_size}  noise={noise_std:.3f}  lr={base_lr:.1e}")
    print(f"{'='*60}\n")

    # reset stopper counter so each phase gets a clean slate
    if early_stopper is not None:
        early_stopper.reset()

    t_transform = build_transform(img_size, noise_std, augment=True)
    v_transform = build_transform(img_size, noise_std, augment=False)

    ds_train = GalaxyZoo(root=dataset_root, mode='train', transform=t_transform)
    ds_val   = GalaxyZoo(root=dataset_root, mode='val',   transform=v_transform)

    if subset is not None and 0 < subset < 1.0:
        N = len(ds_train)
        k = max(1, int(N * subset))
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(N, generator=g)[:k].tolist()
        ds_train = Subset(ds_train, idx)
        print(f"  Subset: {k}/{N} training samples")

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              drop_last=True, pin_memory=True, num_workers=8,
                              collate_fn=GalaxyZoo.collate_fn)
    loader_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                              pin_memory=True, num_workers=8,
                              collate_fn=GalaxyZoo.collate_fn)

    groups = param_groups(model, base_lr, head_lr_scale, early_stage_scale, wd)
    optimizer = torch.optim.AdamW(groups, lr=base_lr, weight_decay=wd)

    steps_per_epoch = math.ceil(len(loader_train) / accum_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[g['lr'] for g in groups],
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=warmup_frac,
    )

    criterion = SoftCrossEntropyLoss()

    phase_best = best_val_acc

    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, loader_train, optimizer, scheduler, criterion,
            device, accum_steps, num_classes, mixup_alpha, ema=ema)

        # evaluate with EMA weights
        ema.apply_to(model)
        val_acc, val_labels, val_probs = evaluate(model, loader_val, device)
        # restore live weights
        model.load_state_dict({k: v.to(next(model.parameters()).dtype)
                               for k, v in ema.shadow.items()})
        # Note: after apply_to + restore, model holds EMA weights
        # for saving we keep them (EMA = best practice)

        elapsed = time.time() - t0

        # early stopping status suffix
        es_status = ""
        if early_stopper is not None:
            es_status = f" | patience {early_stopper.counter}/{early_stopper.patience}"
        print(f"  [{phase_name}] epoch {epoch+1}/{epochs} | "
              f"val_acc={val_acc:.4f} | train_loss={train_loss:.4f} | {elapsed:.0f}s{es_status}")

        # always save latest
        torch.save(model.state_dict(), os.path.join(work_dir, "model_last.pth"))

        if val_acc > phase_best:
            phase_best = val_acc
            torch.save(model.state_dict(), os.path.join(work_dir, "model_best.pth"))
            np.save(os.path.join(work_dir, "best_val_labels.npy"), val_labels)
            np.save(os.path.join(work_dir, "best_val_probs.npy"),  val_probs)
            print(f"  ★ New best val acc: {phase_best:.4f}  (saved)")

        metrics_history.append({
            "phase": phase_name,
            "epoch": epoch + 1,
            "val_acc": val_acc,
            "train_loss": train_loss,
            "img_size": img_size,
            "noise_std": float(noise_std),
            "es_counter": early_stopper.counter if early_stopper else None,
        })
        with open(os.path.join(work_dir, "metrics_history.json"), "w") as f:
            json.dump(metrics_history, f, indent=2)

        # check early stopping — fires AFTER saving so best checkpoint is always kept
        if early_stopper is not None and early_stopper.step(val_acc):
            print(f"  ✗ Early stopping triggered (no improvement for "
                  f"{early_stopper.patience} epochs). Moving to next phase.")
            break

    return phase_best


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="CvT-13 domain-shift adaptation (low-res + noise)")

    # paths
    p.add_argument('--cfg',      default='configs/cvt-13-224x224.yaml', type=str)
    p.add_argument('--pretrain', default='runs/baseline/model_best.pth', type=str,
                   help='Path to your already fine-tuned galaxy weights')
    p.add_argument('--dataset_root', default='data/galaxy_zoo', type=str)
    p.add_argument('--work_dir',     default='runs/domain_adapt', type=str)
    p.add_argument('--run_name',     default=None, type=str)

    # domain-shift targets
    p.add_argument('--target_res',    default=64,  type=int,
                   help='Final resolution to adapt to (e.g. 64 or 32)')
    p.add_argument('--noise_max',     default=0.15, type=float,
                   help='Final Gaussian noise σ to adapt to')

    # curriculum steps (number of intermediate resolutions/noise levels)
    p.add_argument('--noise_steps',   default=2, type=int,
                   help='How many noise levels to ramp through at target_res (0 → noise_max)')
    p.add_argument('--res_steps',     default=3, type=int,
                   help='How many resolution steps from 224 → target_res (snapped to multiples of 16)')
    p.add_argument('--res_values',    default=None, type=int, nargs='+',
                   help='Explicit resolution ladder e.g. --res_values 160 112 96 64. '
                        'Overrides --res_steps / --target_res if provided.')

    # training hyper-params
    p.add_argument('--epochs_per_phase', default=10,    type=int)
    p.add_argument('--epochs_joint',     default=15,    type=int)
    p.add_argument('--batch_size',       default=32,    type=int)
    p.add_argument('--lr',               default=2e-5,  type=float)
    p.add_argument('--wd',               default=0.01,  type=float)
    p.add_argument('--warmup_frac',      default=0.15,  type=float)
    p.add_argument('--accum_steps',      default=4,     type=int)
    p.add_argument('--mixup_alpha',      default=0.4,   type=float,
                   help='Set to 0 to disable mixup')
    p.add_argument('--ema_decay',        default=0.9999, type=float)

    # early stopping
    p.add_argument('--es_patience',  default=5,    type=int,
                   help='Epochs without improvement before stopping a phase (default 5)')
    p.add_argument('--es_min_delta', default=1e-4, type=float,
                   help='Minimum val acc improvement to reset patience counter')

    # differential LR scales
    p.add_argument('--head_lr_scale',        default=10.0, type=float)
    p.add_argument('--early_stage_lr_scale', default=0.1,  type=float)

    # misc
    p.add_argument('--subset',  default=None, type=float)
    p.add_argument('--seed',    default=42, type=int)
    p.add_argument('--device',  default='cuda', type=str)
    p.add_argument('--num_classes', default=5, type=int)

    # distributed (kept for compatibility, not used in single-GPU path)
    p.add_argument('--local_rank', default=0, type=int)
    p.add_argument('--port',       default=9001, type=int)
    p.add_argument('opts', nargs=argparse.REMAINDER, default=None)

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── reproducibility ──────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ── device ───────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CPU mode")

    # ── work dir ─────────────────────────────────────────────────────────
    run_id = args.run_name or f"adapt_{datetime.datetime.now():%Y%m%d-%H%M%S}"
    host   = socket.gethostname().split('.')[0]
    args.work_dir = os.path.join(args.work_dir, f"{run_id}_{host}")
    os.makedirs(args.work_dir, exist_ok=True)
    print(f"Work dir: {args.work_dir}")

    # ── config + model ───────────────────────────────────────────────────
    init_distributed(args)
    setup_cudnn(config)
    update_config(config, args)

    # create_logger mirrors train.py — required by some lib.utils internals
    final_output_dir = create_logger(config, args.cfg, 'train')

    # NUM_CLASSES: the YAML has it commented out so it defaults to 1000.
    # train.py users fix this with --opts MODEL.NUM_CLASSES 5.
    # We also support --num_classes directly here.
    try:
        config.defrost()
        config.MODEL.NUM_CLASSES = args.num_classes
        config.freeze()
        print(f"NUM_CLASSES set to {args.num_classes}")
    except Exception as e:
        print(f"[WARN] Could not override NUM_CLASSES via defrost: {e}")
        print("  Fallback: pass --opts MODEL.NUM_CLASSES 5 on the command line")

    model = build_model(config)

    # load fine-tuned galaxy weights
    state = torch.load(args.pretrain, map_location='cpu')
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded weights from {args.pretrain}")
    if missing:    print(f"  Missing keys  : {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected: print(f"  Unexpected    : {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    model.to(device)

    ema = ModelEMA(model, decay=args.ema_decay)
    early_stopper = EarlyStopping(patience=args.es_patience, min_delta=args.es_min_delta)
    print(f"Early stopping: patience={args.es_patience}, min_delta={args.es_min_delta}")

    # ── curriculum schedules ─────────────────────────────────────────────
    # Resolution steps: use values that divide cleanly through stride 4→2→2
    # (i.e. divisible by 16) so token grids are always integers.
    # Default clean ladder from 224 to 64: 160 → 112 → 64
    # np.linspace is replaced with a fixed-divisor ladder so intermediate
    # steps are always multiples of 16.
    def _res_ladder(start, end, steps):
        """Generate `steps` values from start→end, each rounded to nearest 16."""
        raw = np.linspace(start, end, steps + 1)[1:]   # skip start (already trained on it)
        return [max(16, int(round(r / 16)) * 16) for r in raw]

    if args.res_values is not None:
        res_values = args.res_values
        args.target_res = res_values[-1]
    else:
        res_values = _res_ladder(224, args.target_res, args.res_steps)
    noise_levels = np.linspace(0.0, args.noise_max, args.noise_steps + 1)[1:]  # skip 0

    print(f"\nRes   curriculum : {res_values}")
    print(f"Noise curriculum : {[f'{s:.3f}' for s in noise_levels]}")

    metrics_history = []
    best_val_acc    = -1.0

    # base_lr and epochs are intentionally EXCLUDED from common so they can be
    # overridden cleanly per-phase without risk of duplicate keyword args.
    common = dict(
        dataset_root         = args.dataset_root,
        wd                   = args.wd,
        batch_size           = args.batch_size,
        num_classes          = args.num_classes,
        accum_steps          = args.accum_steps,
        mixup_alpha          = args.mixup_alpha,
        warmup_frac          = args.warmup_frac,
        subset               = args.subset,
        seed                 = args.seed,
        device               = device,
        work_dir             = args.work_dir,
        metrics_history      = metrics_history,
        head_lr_scale        = args.head_lr_scale,
        early_stage_scale    = args.early_stage_lr_scale,
        early_stopper        = early_stopper,
    )

    # ── PHASE 1: resolution curriculum (clean — no noise) ────────────────
    print("\n\n>>> PHASE 1: Resolution Curriculum (no noise, stepping down resolution)")
    for i, res in enumerate(res_values):
        best_val_acc = run_phase(
            phase_name   = f"res_ramp_{i+1}_res{res}",
            img_size     = int(res),
            noise_std    = 0.0,
            base_lr      = args.lr,
            epochs       = args.epochs_per_phase,
            best_val_acc = best_val_acc,
            model        = model,
            ema          = ema,
            **common,
        )

    # ── PHASE 2: noise curriculum (target res fixed) ──────────────────────
    print("\n\n>>> PHASE 2: Noise Curriculum (target res fixed, ramping noise)")
    for i, sigma in enumerate(noise_levels):
        best_val_acc = run_phase(
            phase_name   = f"noise_ramp_{i+1}_sigma{sigma:.3f}",
            img_size     = args.target_res,
            noise_std    = sigma,
            base_lr      = args.lr,
            epochs       = args.epochs_per_phase,
            best_val_acc = best_val_acc,
            model        = model,
            ema          = ema,
            **common,
        )

    # ── PHASE 3: joint fine-tune at target conditions ──────────────────────
    print("\n\n>>> PHASE 3: Joint Fine-Tune (target res + noise_max, lower LR)")
    best_val_acc = run_phase(
        phase_name   = "joint_finetune",
        img_size     = args.target_res,
        noise_std    = args.noise_max,
        base_lr      = args.lr * 0.1,   # consolidation: 10× smaller
        epochs       = args.epochs_joint,
        best_val_acc = best_val_acc,
        model        = model,
        ema          = ema,
        **common,
    )

    total_phases = args.res_steps + args.noise_steps + 1
    print(f"\n{'='*60}")
    print(f"  Domain adaptation complete.")
    print(f"  Total phases run : {total_phases}")
    print(f"  Best val acc     : {best_val_acc:.4f}")
    print(f"  Outputs saved to : {args.work_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()