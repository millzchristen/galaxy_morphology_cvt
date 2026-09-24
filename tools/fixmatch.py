"""
train_fixmatch.py  —  Condition B: FixMatch semi-supervised adaptation
on corrupted Galaxy Zoo data, starting from a finetuned CvT-13 checkpoint.

Corruption applied to BOTH labeled and unlabeled splits matches the
end-of-finetune severity: target_res=64, noise_std=0.15, blue channel × 0.05.

FixMatch loss:
    L = L_s + lambda_u * L_u
where L_u uses pseudo-labels generated from weakly-augmented unlabeled images,
only retained when model confidence >= tau, and enforced on strongly-augmented
versions of the same images.
"""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import time
import csv
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import math

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import _init_paths
from lib.config import config, update_config
from lib.models import build_model
from lib.utils.utils import setup_cudnn, strip_prefix_if_present
from lib.utils.comm import comm

from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as TF
from PIL import ImageOps, ImageFilter

# -----------------------
# Argument parsing
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description='FixMatch domain adaptation - randomized strong augment with blue-shrink op')
    parser.add_argument('--cfg', default='configs/cvt-13-224x224.yaml', type=str)
    parser.add_argument('--run_name', default='fixmatch_run', type=str)
    parser.add_argument('--work_dir', default='runs/fixmatch', type=str)
    parser.add_argument('--pretrain', default=None, type=str, help='path to finetuned checkpoint')
    parser.add_argument('--dataset_root', default='data/imagenet', type=str,
                        help='root containing splits (train/val/test/test_corrupt)')
    parser.add_argument('--data_split', default='test_corrupt', type=str, help='subfolder under dataset_root to use')
    parser.add_argument('--tau', default=0.85, type=float, help='single tau (if --tau_list not given)')
    parser.add_argument('--tau_list', default=None, type=str, help='comma separated taus to sweep e.g. "0.85,0.9,0.95"')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--mu', type=int, default=3, help='unlabeled batch multiplier')
    parser.add_argument('--lambda_u', type=float, default=1.0)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--target_res', type=int, default=64)
    parser.add_argument('--noise_std', type=float, default=0.0, help='Gaussian noise added during augmentation (0.0 to disable).')
    parser.add_argument('--dry_run', action='store_true', help='run a short quick sweep (<=3 epochs)')
    parser.add_argument('--split_adapt', action='store_true', help='(COMMENTED OUT) split data_split into adapt & heldout to avoid leakage')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    # Strong augment pool controls
    parser.add_argument('--num_ops', type=int, default=2, help='number of random ops to apply per strong augmentation')
    parser.add_argument('--magnitude', type=int, default=6, help='magnitude (0..10) for pool ops')
    parser.add_argument('--blue_shrink', type=float, default=0.05, help='blue shrink factor for BlueShrink op (1.0 disables)')
    parser.add_argument('--include_blue_in_pool', action='store_true', help='include BlueShrink op in strong pool')
    args = parser.parse_args()
    return args

# -----------------------
# Utilities, transforms, datasets
# -----------------------
def set_seed(seed):
    import random as pyrand
    import numpy as onp
    pyrand.seed(seed)
    onp.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_normalize():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return T.Normalize(mean=mean, std=std)

# Simple weak augment (returns tensor in [0,1], no blue-shrink)
class WeakAugment:
    def __init__(self, size, noise_std=0.0, model_size=224):
        self.tr = T.Compose([
            T.Resize((size, size)),       # target_res (e.g. 64) — corruption resolution
            T.Resize((model_size, model_size)),  # rescale to model input size (224)
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
        self.noise_std = float(noise_std)

    def __call__(self, img):
        t = self.tr(img)
        if self.noise_std and self.noise_std > 0.0:
            t = t + torch.randn_like(t) * self.noise_std
            t = torch.clamp(t, 0.0, 1.0)
        return t

# BlueShrink as PIL op (used in pool)
class BlueShrinkPIL:
    def __init__(self, factor=0.05):
        self.factor = float(factor)

    def __call__(self, img):
        arr = np.array(img).astype(np.float32)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            arr[:, :, 2] = arr[:, :, 2] * self.factor
            arr = np.clip(arr, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))

# Randomized strong augmentation pool (PIL ops -> ToTensor -> optional noise)
class StrongAugmentRand:
    def __init__(self, size, num_ops=2, magnitude=6, noise_std=0.0, blue_shrink=0.05,
                 include_blue=True, model_size=224):
        self.size = int(size)
        self.model_size = int(model_size)
        self.num_ops = int(num_ops)
        self.magnitude = int(magnitude)
        self.noise_std = float(noise_std)
        self.blue_shrink = float(blue_shrink)
        self.include_blue = bool(include_blue)
        self.pre = T.RandomResizedCrop(self.size, scale=(0.2, 1.0))
        # Always rescale to model input size after crop
        self.resize_to_model = T.Resize((self.model_size, self.model_size))
        self.post_to_tensor = T.ToTensor()

    def int_parameter(self, level, maxval):
        return int(level * maxval / 10)

    def float_parameter(self, level, maxval):
        return float(level) * float(maxval) / 10.0

    # pool op implementations (PIL in -> PIL out)
    def _rotate(self, img, magnitude):
        degrees = self.int_parameter(magnitude, 30)
        angle = random.uniform(-degrees, degrees)
        return img.rotate(angle)

    def _shear_x(self, img, magnitude):
        level = self.float_parameter(magnitude, 0.3)
        if random.random() < 0.5: level = -level
        return img.transform(img.size, Image.AFFINE, (1, level, 0, 0, 1, 0))

    def _shear_y(self, img, magnitude):
        level = self.float_parameter(magnitude, 0.3)
        if random.random() < 0.5: level = -level
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, level, 1, 0))

    def _translate_x(self, img, magnitude):
        max_shift = self.int_parameter(magnitude, int(0.45 * img.size[0]))
        if random.random() < 0.5: max_shift = -max_shift
        return img.transform(img.size, Image.AFFINE, (1, 0, max_shift, 0, 1, 0))

    def _translate_y(self, img, magnitude):
        max_shift = self.int_parameter(magnitude, int(0.45 * img.size[1]))
        if random.random() < 0.5: max_shift = -max_shift
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, max_shift))

    def _color_jitter(self, img, magnitude):
        b = self.float_parameter(magnitude, 0.8)
        c = self.float_parameter(magnitude, 0.8)
        s = self.float_parameter(magnitude, 0.8)
        h = self.float_parameter(magnitude, 0.2)
        out = TF.adjust_brightness(img, b)
        out = TF.adjust_contrast(out, c)
        out = TF.adjust_saturation(out, s)
        out = TF.adjust_hue(out, h)
        return out

    def _autocontrast(self, img, magnitude): return ImageOps.autocontrast(img)
    def _invert(self, img, magnitude): return ImageOps.invert(img)
    def _equalize(self, img, magnitude): return ImageOps.equalize(img)
    def _solarize(self, img, magnitude):
        thresh = self.int_parameter(magnitude, 256)
        return ImageOps.solarize(img, 256 - thresh)
    def _posterize(self, img, magnitude):
        bits = max(1, 8 - self.int_parameter(magnitude, 7))
        return ImageOps.posterize(img, bits)
    def _gaussian_blur(self, img, magnitude):
        r = self.float_parameter(magnitude, 2.0)
        return img.filter(ImageFilter.GaussianBlur(radius=r))
    def _sharpness(self, img, magnitude):
        from PIL import ImageEnhance
        factor = 1.0 + self.float_parameter(magnitude, 1.5)
        return ImageEnhance.Sharpness(img).enhance(factor)
    def _brightness(self, img, magnitude):
        from PIL import ImageEnhance
        factor = 1.0 + self.float_parameter(magnitude, 0.9)
        return ImageEnhance.Brightness(img).enhance(factor)
    def _contrast(self, img, magnitude):
        from PIL import ImageEnhance
        factor = 1.0 + self.float_parameter(magnitude, 0.9)
        return ImageEnhance.Contrast(img).enhance(factor)

    def _blue_shrink_pil(self, img):
        if self.blue_shrink == 1.0:
            return img
        op = BlueShrinkPIL(self.blue_shrink)
        return op(img)

    def _op_pool(self):
        pool = [
            ("rotate", self._rotate),
            ("shear_x", self._shear_x),
            ("shear_y", self._shear_y),
            ("translate_x", self._translate_x),
            ("translate_y", self._translate_y),
            ("color_jitter", self._color_jitter),
            ("autocontrast", self._autocontrast),
            ("equalize", self._equalize),
            ("invert", self._invert),
            ("solarize", self._solarize),
            ("posterize", self._posterize),
            ("gaussian_blur", self._gaussian_blur),
            ("sharpness", self._sharpness),
            ("brightness", self._brightness),
            ("contrast", self._contrast),
        ]
        if self.include_blue and self.blue_shrink != 1.0:
            pool.append(("blue_shrink", lambda img, mag: self._blue_shrink_pil(img)))
        return pool

    def __call__(self, img):
        img = self.pre(img)          # RandomResizedCrop → target_res (e.g. 64)
        pool = self._op_pool()
        ops = random.sample(pool, k=min(self.num_ops, len(pool)))
        out = img
        for name, fn in ops:
            try:
                out = fn(out, self.magnitude)
            except TypeError:
                out = fn(out)
        out = self.resize_to_model(out)   # rescale to 224×224 for model input
        t = self.post_to_tensor(out)
        if self.noise_std and self.noise_std > 0:
            t = t + torch.randn_like(t) * float(self.noise_std)
            t = torch.clamp(t, 0.0, 1.0)
        return t

# Unlabeled pair dataset wraps an ImageFolder to return (weak_tensor, strong_tensor)
class UnlabeledPairDataset(Dataset):
    def __init__(self, base_imagefolder, weak_transform, strong_transform, normalize=None):
        self.base = base_imagefolder
        self.weak = weak_transform
        self.strong = strong_transform
        self.normalize = normalize

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]  # PIL.Image, label ignored
        w = self.weak(img)       # tensor [C,H,W]
        s = self.strong(img)     # tensor [C,H,W]
        if self.normalize is not None:
            w = self.normalize(w)
            s = self.normalize(s)
        return w, s

# -----------------------
# Training / evaluation functions
# -----------------------
def train_one_epoch(model, labeled_loader, unlabeled_loader, optimizer, device, epoch, args, tau):
    model.train()
    criterion = torch.nn.CrossEntropyLoss().to(device)

    total_loss = 0.0
    total_x_loss = 0.0
    total_u_loss = 0.0
    total_steps = 0
    mask_counts = 0
    mask_total = 0

    # CASE A: have labeled loader (rare in our setup) -> iterate labeled batches paired with unlabeled
    if labeled_loader is not None:
        unlabeled_iter = iter(unlabeled_loader)
        pbar = tqdm(enumerate(labeled_loader), total=len(labeled_loader),
                    desc=f"Epoch {epoch} (tau={tau})", leave=False)
        for batch_idx, batch in pbar:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x_l, y_l = batch[0], batch[1]
            elif isinstance(batch, dict):
                x_l = batch.get('image') or batch.get('img') or batch.get('images')
                y_l = batch.get('label') or batch.get('target') or batch.get('labels')
            else:
                raise RuntimeError("Unexpected labeled batch format")

            try:
                x_ul_w, x_ul_s = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                x_ul_w, x_ul_s = next(unlabeled_iter)

            x_l = x_l.to(device); y_l = y_l.to(device).long()
            x_ul_w = x_ul_w.to(device); x_ul_s = x_ul_s.to(device)

            logits_x = model(x_l)
            if isinstance(logits_x, (list, tuple)): logits_x = logits_x[0]
            loss_x = criterion(logits_x, y_l)

            with torch.no_grad():
                out_w = model(x_ul_w)
                if isinstance(out_w, (list, tuple)): out_w = out_w[0]
                probs = F.softmax(out_w, dim=1)
                max_probs, p_hat = probs.max(dim=1)
                mask = max_probs.ge(tau).float()
                mask_counts += int(mask.sum().item())
                mask_total += mask.numel()

            out_s = model(x_ul_s)
            if isinstance(out_s, (list, tuple)): out_s = out_s[0]
            loss_u_all = F.cross_entropy(out_s, p_hat, reduction='none')
            if mask.sum().item() > 0:
                loss_u = (loss_u_all * mask).mean()
            else:
                loss_u = loss_u_all.sum() * 0.0

            loss = loss_x + args.lambda_u * loss_u

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_x_loss += float(loss_x.item())
            total_u_loss += float(loss_u.item()) if isinstance(loss_u, torch.Tensor) else float(loss_u)
            total_steps += 1

            if batch_idx % args.print_freq == 0:
                cur_mask_rate = mask_counts / (mask_total + 1e-12)
                pbar.set_postfix({
                    'loss': f"{total_loss / total_steps:.4f}",
                    'lx': f"{total_x_loss / total_steps:.4f}",
                    'lu': f"{(total_u_loss / max(1, total_steps)):.4f}",
                    'mask_rate': f"{cur_mask_rate:.4f}"
                })

    # CASE B: no labeled loader -> iterate directly over unlabeled_loader (pseudo-label-only updates)
    else:
        pbar = tqdm(enumerate(unlabeled_loader), total=len(unlabeled_loader),
                    desc=f"Epoch {epoch} (tau={tau})", leave=False)
        for batch_idx, batch in pbar:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x_ul_w, x_ul_s = batch
            else:
                raise RuntimeError("Unexpected unlabeled batch format; expected (weak_batch, strong_batch)")

            x_ul_w = x_ul_w.to(device); x_ul_s = x_ul_s.to(device)

            with torch.no_grad():
                out_w = model(x_ul_w)
                if isinstance(out_w, (list, tuple)): out_w = out_w[0]
                probs = F.softmax(out_w, dim=1)
                max_probs, p_hat = probs.max(dim=1)
                mask = max_probs.ge(tau).float()
                mask_counts += int(mask.sum().item())
                mask_total += mask.numel()

            out_s = model(x_ul_s)
            if isinstance(out_s, (list, tuple)): out_s = out_s[0]
            loss_u_all = F.cross_entropy(out_s, p_hat, reduction='none')
            if mask.sum().item() > 0:
                loss_u = (loss_u_all * mask).mean()
            else:
                loss_u = loss_u_all.sum() * 0.0

            loss = args.lambda_u * loss_u

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_u_loss += float(loss_u.item()) if isinstance(loss_u, torch.Tensor) else float(loss_u)
            total_steps += 1

            if batch_idx % args.print_freq == 0:
                cur_mask_rate = mask_counts / (mask_total + 1e-12)
                pbar.set_postfix({
                    'loss': f"{total_loss / total_steps:.4f}",
                    'lu': f"{(total_u_loss / max(1, total_steps)):.4f}",
                    'mask_rate': f"{cur_mask_rate:.4f}"
                })

    epoch_mask_rate = mask_counts / (mask_total + 1e-12)
    avg_loss = total_loss / max(1, total_steps)
    return avg_loss, epoch_mask_rate

def evaluate_and_save(model, eval_loader, device, out_dir, epoch_num):
    model.eval()
    all_labels = []
    all_preds = []
    all_logits = []
    
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluation", leave=False):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images, labels = batch[0], batch[1]
            elif isinstance(batch, dict):
                images = batch.get('image') or batch.get('img') or batch.get('images')
                labels = batch.get('label') or batch.get('target') or batch.get('labels')
            else:
                images, labels = batch
            images = images.to(device)
            labels = labels.to(device).long()

            output = model(images)
            if isinstance(output, (list, tuple)): output = output[0]
            _, pred = output.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_logits.extend(output.cpu().numpy())

            correct += int((pred.cpu() == labels.cpu()).sum().item())
            total += labels.size(0)

    labels_fname = os.path.join(out_dir, f"test_labels_e{epoch_num}.npy")
    preds_fname = os.path.join(out_dir, f"test_preds_e{epoch_num}.npy")
    logits_fname = os.path.join(out_dir, f"test_logits_e{epoch_num}.npy")

    np.save(labels_fname, np.array(all_labels))
    np.save(preds_fname, np.array(all_preds))
    np.save(logits_fname, np.array(all_logits)) 
    acc = correct / total if total > 0 else 0.0
    logging.info(f"=> Eval epoch {epoch_num}: saved labels to {labels_fname} preds to {preds_fname} logits to {logits_fname} acc={acc:.4f}")
    return acc

# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    update_config(config, args)
    final_output_dir = os.path.join(args.work_dir, args.run_name)
    os.makedirs(final_output_dir, exist_ok=True)

    if comm.is_main_process():
        logging.basicConfig(level=logging.INFO)
        logging.info("Config:")
        logging.info(config)

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    setup_cudnn(config)

    if args.dry_run:
        logging.info("Dry run enabled -> limiting epochs to <=3")
        args.epochs = min(3, args.epochs)

    # parse tau list
    if args.tau_list:
        tau_values = [float(x) for x in args.tau_list.split(',')]
    else:
        tau_values = [args.tau]

    # ---------------- COMMENTED OUT: optional split of data_split into adapt & heldout ----------------
    # if args.split_adapt:
    #     # Example: split dataset_root/data_split into adapt_unlabeled/ and heldout_eval/ to avoid leakage.
    #     # Implement copying or symlinking preserving class folders as needed.
    #     pass
    # ----------------------------------------------------------------------------------------------

    # Build evaluation loader from dataset_root/<data_split> (ImageFolder). Use val transforms (no augmentation).
    data_dir = os.path.join(args.dataset_root, args.data_split)
    if not os.path.isdir(data_dir):
        raise RuntimeError(f"Data split directory not found at expected path: {data_dir}")

    normalize = build_normalize()
    val_transform = T.Compose([
        T.Resize((args.target_res, args.target_res)),  # match corruption resolution
        T.Resize((224, 224)),                           # rescale to model input size
        T.ToTensor(),
        normalize
    ])

    eval_dataset = ImageFolder(data_dir, transform=val_transform)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=max(2, args.num_workers), pin_memory=True)

    # Build unlabeled pair dataset from same data_dir (treat as unlabeled)
    weak_t = WeakAugment(args.target_res, noise_std=args.noise_std, model_size=224)
    strong_t = StrongAugmentRand(size=args.target_res, num_ops=args.num_ops, magnitude=args.magnitude,
                                 noise_std=args.noise_std, blue_shrink=args.blue_shrink,
                                 include_blue=True, model_size=224)

    base_imagefolder = ImageFolder(data_dir)  # no transform; returns PIL images
    unlabeled_pair_dataset = UnlabeledPairDataset(base_imagefolder, weak_transform=weak_t,
                                                  strong_transform=strong_t, normalize=normalize)
    unlabeled_loader = DataLoader(unlabeled_pair_dataset, batch_size=args.batch_size * args.mu,
                                  shuffle=True, num_workers=max(2, args.num_workers),
                                  pin_memory=True, drop_last=True)

    # labeled_loader: None by default (we perform pseudo-label-only adaptation). If you have labeled set, construct similarly.
    labeled_loader = None

    # Build model and run tau sweeps
    model = build_model(config)
    model.to(device)

    for tau in tau_values:
        tau_str = f"{tau:.2f}".replace('.', '_')
        run_dir = os.path.join(final_output_dir, f"tau_{tau_str}")
        os.makedirs(run_dir, exist_ok=True)
        logging.info(f"Starting run for tau={tau} -> {run_dir}")

        # reload pretrained checkpoint to ensure same init for each tau
        if args.pretrain:
            logging.info(f"=> loading pretrained weights from: {args.pretrain}")
            state = torch.load(args.pretrain, map_location='cpu')
            if isinstance(state, dict) and 'state_dict' in state:
                state_dict = strip_prefix_if_present(state['state_dict'], 'module.')
            else:
                state_dict = state
            model.load_state_dict(state_dict, strict=False)
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
        tb_writer = SummaryWriter(log_dir=run_dir)

        mask_rates = []
        val_accs = []

        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss, epoch_mask_rate = train_one_epoch(model, labeled_loader, unlabeled_loader, optimizer, device, epoch, args, tau)
            mask_rates.append(epoch_mask_rate)

            # evaluate on the evaluation loader and save preds/labels per epoch
            val_acc = evaluate_and_save(model, eval_loader, device, run_dir, epoch)
            val_accs.append(val_acc)

            tb_writer.add_scalar('train/loss', train_loss, epoch)
            tb_writer.add_scalar('train/mask_rate', epoch_mask_rate, epoch)
            tb_writer.add_scalar('val/acc', val_acc, epoch)

            logging.info(f"Epoch {epoch} done in {time.time() - start:.1f}s loss={train_loss:.4f} mask_rate={epoch_mask_rate:.4f} val_acc={val_acc:.4f}")

            # save checkpoint
            ckpt_path = os.path.join(run_dir, f"model_epoch{epoch}.pth")
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, ckpt_path)

        # final save & summaries
        final_ckpt = os.path.join(run_dir, f"model_final.pth")
        torch.save({'state_dict': model.state_dict(), 'epochs': args.epochs}, final_ckpt)
        logging.info(f"Saved final model to {final_ckpt}")

        np.save(os.path.join(run_dir, "mask_rates.npy"), np.array(mask_rates))
        np.save(os.path.join(run_dir, "val_accs.npy"), np.array(val_accs))
        csv_fpath = os.path.join(run_dir, "mask_rates.csv")
        with open(csv_fpath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['epoch', 'mask_rate', 'val_acc'])
            for e, (m, a) in enumerate(zip(mask_rates, val_accs), start=1):
                writer.writerow([e, m, a])

        tb_writer.close()
        logging.info(f"Finished tau={tau}. Mask rates saved to {run_dir}/mask_rates.npy and {csv_fpath}")

    logging.info("All tau runs finished.")

if __name__ == "__main__":
    main()