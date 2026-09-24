from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import pickle as pkl
import pprint
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
from torch.utils.collect_env import get_pretty_env_info
from torch.utils.tensorboard import SummaryWriter

import _init_paths
from config import config
from config import update_config
from core.function import test
from core.loss import build_criterion
from dataset import build_dataloader
from dataset import RealLabelsImagenet
from models import build_model
from utils.comm import comm
from utils.utils import create_logger
from utils.utils import init_distributed
from utils.utils import setup_cudnn
from utils.utils import summary_model_on_master
from utils.utils import strip_prefix_if_present


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test classification network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='configs/cvt-13-224x224.yaml',
                        type=str)

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--port", type=int, default=9000)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--run_name', default=None, type=str,
                        help='Optional name for this run (overrides SLURM_JOB_NAME)')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    init_distributed(args)
    setup_cudnn(config)

    update_config(config, args)
    final_output_dir = create_logger(config, args.cfg, 'test')

    if args.run_name:
        final_output_dir = os.path.join(final_output_dir, args.run_name)
        os.makedirs(final_output_dir, exist_ok=True)

    tb_log_dir = final_output_dir

    if comm.is_main_process():
        logging.info("=> collecting env info (might take some time)")
        logging.info("\n" + get_pretty_env_info())
        logging.info(pprint.pformat(args))
        logging.info(config)
        logging.info("=> using {} GPUs".format(args.num_gpus))

        output_config_path = os.path.join(final_output_dir, 'config.yaml')
        logging.info("=> saving config into: {}".format(output_config_path))

    model = build_model(config)
    model.to(torch.device('cuda'))

    model_file = config.TEST.MODEL_FILE if config.TEST.MODEL_FILE \
        else os.path.join(final_output_dir, 'model_best.pth')
    logging.info('=> load model file: {}'.format(model_file))
    ext = model_file.split('.')[-1]
    if ext == 'pth':
        state_dict = torch.load(model_file, map_location="cpu")
    else:
        raise ValueError("Unknown model file")

    model.load_state_dict(state_dict, strict=False)
    model.to(torch.device('cuda'))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    summary_model_on_master(model, config, final_output_dir, False)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    # define loss function (criterion) and optimizer
    criterion = build_criterion(config, train=False)
    criterion.cuda()

    valid_loader = build_dataloader(config, False, args.distributed)

    # YAML resize check - print one sample batch image shape
    if comm.is_main_process():
        try:
            sample_batch = next(iter(valid_loader))
            if isinstance(sample_batch, (list, tuple)) and len(sample_batch) >= 1:
                sample_images = sample_batch[0]
            elif isinstance(sample_batch, dict):
                if 'image' in sample_batch:
                    sample_images = sample_batch['image']
                elif 'img' in sample_batch:
                    sample_images = sample_batch['img']
                elif 'images' in sample_batch:
                    sample_images = sample_batch['images']
                else:
                    sample_images = next(iter(sample_batch.values()))
            else:
                raise RuntimeError("Unknown batch format for size check")

            logging.info(f"Sample batch image tensor shape: {sample_images.shape}")
            print(f"\n>>> YAML Resize Check: Input image shape = {sample_images.shape}\n")
        except Exception as e:
            logging.warning(f"Failed to inspect input image size: {e}")

    real_labels = None
    if (
        config.DATASET.DATASET == 'imagenet'
        and config.DATASET.DATA_FORMAT == 'tsv'
        and config.TEST.REAL_LABELS
    ):
        filenames = valid_loader.dataset.get_filenames()
        real_json = os.path.join(config.DATASET.ROOT, 'real.json')
        logging.info('=> loading real labels...')
        real_labels = RealLabelsImagenet(filenames, real_json)

    valid_labels = None
    if config.TEST.VALID_LABELS:
        with open(config.TEST.VALID_LABELS, 'r') as f:
            valid_labels = {
                int(line.rstrip()) for line in f
            }
            valid_labels = [
                i in valid_labels for i in range(config.MODEL.NUM_CLASSES)
            ]

    logging.info('=> start testing')
    start = time.time()
    
    test(config, valid_loader, model, criterion,
         final_output_dir, tb_log_dir, writer_dict,
         args.distributed, real_labels=real_labels,
         valid_labels=valid_labels)
    
    logging.info('=> test duration time: {:.2f}s'.format(time.time() - start))

    import shutil
    from tqdm import tqdm

    model.eval()
    device = next(model.parameters()).device

    logging.info('=> [Galaxy Zoo] start labeled evaluation')

    epoch_num = 0
    if hasattr(config.TEST, 'EPOCH') and config.TEST.EPOCH is not None:
        try:
            epoch_num = int(config.TEST.EPOCH)
        except Exception:
            epoch_num = 0

    print_freq = getattr(config, "PRINT_FREQ", 50)
    all_labels_gz = []
    all_preds_gz = []
    all_logits_gz = [] 

    with torch.no_grad():
        total, correct, cnt = 0, 0, 0
        for batch_id, batch in enumerate(tqdm(valid_loader, desc="[Galaxy Zoo] Evaluating"), start=1):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images, labels = batch[0], batch[1]
            elif isinstance(batch, dict):
                labels = batch.get('label') or batch.get('target') or batch.get('labels')
                images = batch.get('image') or batch.get('img') or batch.get('images')
                if labels is None:
                    raise RuntimeError("Could not find labels key in batch dict.")
                if images is None:
                    raise RuntimeError("Could not find images key in batch dict.")
            else:
                raise RuntimeError("Unknown batch type from dataloader.")

            images = images.to(device)
            labels = labels.to(device).long()

            output = model(images)
            logits = output[0] if isinstance(output, (list, tuple)) else output

            loss = criterion(logits, labels)
            _, pred = logits.max(1)

            all_labels_gz.extend(labels.detach().cpu().numpy())
            all_preds_gz.extend(pred.detach().cpu().numpy())
            all_logits_gz.extend(logits.detach().cpu().numpy())

            total += labels.size(0)
            correct += int((pred.detach().cpu() == labels.detach().cpu()).sum().item())
            accuracy = correct / total

            cnt += 1
            if cnt % print_freq == 0:
                print(f'  batch {batch_id}/{len(valid_loader)}  '
                      f'loss: {loss.item():.3f}  accuracy: {accuracy:.3f}')

    labels_fname = os.path.join(final_output_dir, f"test_labels_e{epoch_num}.npy")
    preds_fname  = os.path.join(final_output_dir, f"test_preds_e{epoch_num}.npy")
    logits_fname = os.path.join(final_output_dir, f"test_logits_e{epoch_num}.npy")
    np.save(labels_fname, np.array(all_labels_gz))
    np.save(preds_fname,  np.array(all_preds_gz))
    np.save(logits_fname, np.array(all_logits_gz))
    logging.info(f'=> [Galaxy Zoo] Labels saved to: {labels_fname}')
    logging.info(f'=> [Galaxy Zoo] Preds  saved to: {preds_fname}')
    logging.info(f'=> [Galaxy Zoo] Logits saved to: {logits_fname}')
    logging.info(f'=> [Galaxy Zoo] Final accuracy: {correct}/{total} = {correct/total:.4f}')


    writer_dict['writer'].close()
    logging.info('=> finish testing')


if __name__ == '__main__':
    main()
