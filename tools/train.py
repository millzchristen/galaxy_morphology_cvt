from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import pprint
import time
import torch
import torch.nn.parallel
import torch.optim as optim
from torch import nn
from torch.utils.collect_env import get_pretty_env_info
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as tfs
import torch.optim.lr_scheduler as lr_scheduler

import _init_paths
from lib.config import config
from lib.config import update_config
from lib.config import save_config
from lib.models import build_model
from lib.dataset.galaxy_zoo import GalaxyZoo
from lib.utils.comm import comm
from lib.utils.utils import create_logger
from lib.utils.utils import init_distributed
from lib.utils.utils import setup_cudnn

# added
import numpy as np
import datetime
import socket
import random
import math
import json
import shutil # ADDED: For copying config file

### ADDED - GAUSSIAN NOISE
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        if self.std <= 0: return tensor
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
        
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train classification network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='configs/cvt-13-224x224.yaml',
                        type=str)
    parser.add_argument('--dataset_root',
                        default='data/galaxy_zoo',
                        type=str,
                        help='Galaxy Zoo ImageFolder root (train/ and val/)')

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=50, type=int)

    # tuning params
    parser.add_argument('--lr', default=1e-4, type=float, help='override learning rate')
    parser.add_argument('--wd', default=None, type=float, help='override weight decay')
    parser.add_argument('--subset', default=None, type=float,
                        help='fraction of training data to use (e.g. 0.1)')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--noise_std', default=0.0, type=float)

    parser.add_argument('--pretrain',
                        default="pretrain/CvT-13-224x224-IN-1k.pth",
                        type=str)
    parser.add_argument('--work_dir', default="runs", type=str)

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--port", type=int, default=9000)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    
    # added for saving runs by name
    parser.add_argument('--run_name', default=None, type=str,
                        help='Optional name for this run (overrides SLURM_JOB_NAME)')

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- SAFE DEVICE SELECTION ----
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda:0")
            gpu_index = torch.cuda.current_device()
            print(f"Using GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}")
            torch.cuda.empty_cache()
        except Exception as e:
            print("CUDA appears available but failed to initialize.")
            print("Error:", e)
            print("Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    ### reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # run naming
    if args.run_name:
        run_id = args.run_name
    else:
        run_id = os.environ.get("SLURM_JOB_NAME") \
                 or os.environ.get("SLURM_JOB_ID") \
                 or f"run_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{os.getpid()}"

    host = socket.gethostname().split('.')[0]
    run_dirname = f"{run_id}_{host}"
    args.work_dir = os.path.join(args.work_dir, run_dirname)
    os.makedirs(args.work_dir, exist_ok=True)

    print(f"Using per-job work_dir: {args.work_dir}")
    
    ### config and env
    init_distributed(args)
    setup_cudnn(config)
    update_config(config, args)

    # MODIFIED: Copy the config file specifically to "config.yaml" inside work_dir 
    # so the analyze script can find it regardless of the original filename
    try:
        shutil.copy(args.cfg, os.path.join(args.work_dir, "config.yaml"))
    except Exception as e:
        print(f"[WARN] Could not copy config to work_dir: {e}")

    try:
        need_override = (getattr(args, "lr", None) is not None) or (getattr(args, "wd", None) is not None)
        if need_override:
            config.defrost()
            if getattr(args, "lr", None) is not None:
                config.TRAIN.LR = float(args.lr)
            if getattr(args, "wd", None) is not None:
                config.TRAIN.WD = float(args.wd)
            config.freeze()
    except Exception as e:
        print(f"[ERROR] Failed to apply CLI overrides to config: {e}")
        raise

    final_output_dir = create_logger(config, args.cfg, 'train')

    if comm.is_main_process():
        logging.info(get_pretty_env_info())
        logging.info(pprint.pformat(args))
        logging.info(config)
        output_config_path = os.path.join(final_output_dir, 'config.yaml')
        save_config(config, output_config_path)

    device = torch.device(args.device)

    ### model
    model = build_model(config)
    state_dict = torch.load(args.pretrain, map_location="cpu")
    for key in ["head.weight", "head.bias"]:
        if key in state_dict:
            del state_dict[key]
    print(model.load_state_dict(state_dict, strict=False))
    model.to(device)

    ### data
    transform = {
        "train": tfs.Compose([
            tfs.Resize([224, 224]),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            AddGaussianNoise(0., args.noise_std)
        ]),
        "val": tfs.Compose([
            tfs.Resize([224, 224]),
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    
    dataset_train = GalaxyZoo(root=args.dataset_root, mode='train', transform=transform["train"])
    dataset_val = GalaxyZoo(root=args.dataset_root, mode='val', transform=transform["val"])

    if args.subset is not None and 0.0 < args.subset < 1.0:
        N = len(dataset_train)
        k = max(1, int(N * args.subset))
        indices = torch.randperm(N)[:k].tolist()
        dataset_train = Subset(dataset_train, indices)
        print(f"[INFO] Using {k}/{N} samples ({args.subset*100:.1f}%)")
    
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=8, collate_fn=GalaxyZoo.collate_fn)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8, collate_fn=GalaxyZoo.collate_fn)

    # optimizer
    opt_name = str(config.TRAIN.OPTIMIZER).lower()
    base_lr = float(config.TRAIN.LR)
    weight_decay = float(config.TRAIN.WD)

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        print(f"Using AdamW (lr={base_lr}, wd={weight_decay})")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
        print(f"Using Adam (lr={base_lr})")

    criterion = nn.CrossEntropyLoss()
    accum_steps = 8
    optimizer.zero_grad()

    # OneCycleLR automatically handles warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.TRAIN.LR,
        steps_per_epoch=math.ceil(len(dataloader_train) / accum_steps), 
        epochs=args.epochs,
        pct_start=(args.warmup_epochs / args.epochs) 
    )

    print_freq = 50
    best_val_acc = -1.0
    metrics_history = []
    metrics_history_path = os.path.join(args.work_dir, "metrics_history.json")

    ### training
    training_start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        total, correct = 0, 0
        train_losses = []
        train_epoch_data = {"labels": [], "preds": [], "probs": []}

        for batch_id, (images, labels) in enumerate(dataloader_train, start=1):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels) / accum_steps
            loss.backward()

            if batch_id % accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # MODIFIED: Get current LR to display warmup progress
            current_lr = optimizer.param_groups[0]['lr']

            _, pred = output.max(1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            
            probs = torch.softmax(output, dim=1)
            train_epoch_data["labels"].extend(labels.detach().cpu().numpy())
            train_epoch_data["preds"].extend(pred.detach().cpu().numpy())
            train_epoch_data["probs"].extend(probs.detach().cpu().numpy())
            train_losses.append(loss.item() * accum_steps)

            # MODIFIED: Added LR to the print output to monitor warmup
            if batch_id % (print_freq * accum_steps) == 0:
                print(f'Epoch[{epoch}/{args.epochs}] Batch[{batch_id}/{len(dataloader_train)}] '
                      f'Loss: {loss.item()*accum_steps:.3f} Accuracy: {correct/total:.3f} '
                      f'LR: {current_lr:.2e}')

        if batch_id % accum_steps != 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Hierarchial Model: training for 

        # ADDED: Track training accuracy per epoch
        avg_train_acc = correct / total

        ### Validation
        model.eval()
        val_total, val_correct = 0, 0
        val_losses = [] # ADDED: Track validation loss
        val_epoch_data = {"labels": [], "preds": [], "probs": []}

        with torch.no_grad():
            for images, labels in dataloader_val:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                
                # ADDED: Calculate validation loss
                v_loss = criterion(output, labels)
                val_losses.append(v_loss.item())

                probs = torch.softmax(output, dim=1)
                _, pred = output.max(1)
                val_epoch_data["labels"].extend(labels.cpu().numpy())
                val_epoch_data["preds"].extend(pred.cpu().numpy())
                val_epoch_data["probs"].extend(probs.cpu().numpy())
                val_total += labels.size(0)
                val_correct += (pred == labels).sum().item()

        val_acc = val_correct / val_total
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses) # ADDED: Average validation loss

        print(f"--- Epoch {epoch} | Val Acc: {val_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Train Loss: {avg_train_loss:.4f} ---")

        # --- SAVE LOGIC ---
        torch.save(model.state_dict(), os.path.join(args.work_dir, "model_last.pth"))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New Best Accuracy! Saving weights and arrays...")
            torch.save(model.state_dict(), os.path.join(args.work_dir, "model_best.pth"))
            np.save(os.path.join(args.work_dir, "best_val_labels.npy"), np.array(val_epoch_data["labels"]))
            np.save(os.path.join(args.work_dir, "best_val_probs.npy"), np.array(val_epoch_data["probs"]))
            np.save(os.path.join(args.work_dir, "best_train_probs.npy"), np.array(train_epoch_data["probs"]))
            # MODIFIED: Save training labels so training CM can be generated
            np.save(os.path.join(args.work_dir, "best_train_labels.npy"), np.array(train_epoch_data["labels"]))

        # MODIFIED: Append all 4 core metrics for complete plotting
        metrics_history.append({
            "epoch": epoch, 
            "train_loss": avg_train_loss,
            "train_acc": avg_train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc
        })
        
        with open(metrics_history_path, "w") as f:
            json.dump(metrics_history, f)

    total_time = time.time() - training_start_time
    print(f"Total training time: {total_time/3600:.2f} hours. Best Val Acc: {best_val_acc:.4f}")
    
if __name__ == '__main__':
    main()