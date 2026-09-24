from __future__ import absolute_import, division, print_function
 
import argparse
import logging
import os
import csv
import shutil
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
 
import _init_paths
from lib.config import config, update_config
from lib.models import build_model
from lib.utils.utils import setup_cudnn, strip_prefix_if_present
 
from torchvision import transforms as T
 
 
def parse_args():
    parser = argparse.ArgumentParser(description='JADES inference with feature extraction')
    parser.add_argument('--cfg', required=True, type=str, help='path to config yaml')
    parser.add_argument('--checkpoint', required=True, type=str, help='path to trained FixMatch checkpoint')
    parser.add_argument('--jades_dir', required=True, type=str, help='directory with MASKED JADES cutouts')
    parser.add_argument('--original_dir', required=True, type=str, help='directory with ORIGINAL JADES cutouts (for viewer)')
    parser.add_argument('--metadata_file', required=True, type=str, help='path to cutout_metadata.json')
    parser.add_argument('--output_dir', required=True, type=str, help='output directory for organized predictions')
    parser.add_argument('--features_output', required=True, type=str, help='output .npz file for features + predictions')
    parser.add_argument('--tau', type=float, default=0.85, help='confidence threshold')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for inference')
    parser.add_argument('--num_workers', type=int, default=8, help='number of dataloader workers')
    parser.add_argument('--device', default='cuda', type=str, help='device to use')
    parser.add_argument('--target_res', type=int, default=64, help='resolution images were corrupted to')
    parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args
 
 
class JADESInferenceDataset(Dataset):
    
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        self.image_files = sorted([
            f for f in os.listdir(image_dir) 
            if f.lower().endswith('.jpg')
        ])
        
        if len(self.image_files) == 0:
            raise RuntimeError(f"No JPG files found in {image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        filepath = os.path.join(self.image_dir, filename)
        
        # Load image
        img = Image.open(filepath).convert('RGB')
        
        # Apply transform
        if self.transform is not None:
            img = self.transform(img)
        
        return img, filename
 

class FeatureExtractor:
    
    def __init__(self, model):
        self.features = None
        self.model = model
        
        # Register hook on the layer before the classification head
        if hasattr(model, 'norm'):
            model.norm.register_forward_hook(self.hook_fn)
        elif hasattr(model, 'head'):
            layers = list(model.children())
            if len(layers) > 1:
                layers[-2].register_forward_hook(self.hook_fn)
        else:
            raise RuntimeError("Could not find appropriate layer to extract features from")
    
    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        
        # If output is [batch, seq_len, embed_dim], take mean over seq_len
        if output.dim() == 3:
            output = output.mean(dim=1)
        elif output.dim() == 2:
            pass
        else:
            raise RuntimeError(f"Unexpected feature shape: {output.shape}")
        
        self.features = output.detach().cpu()
    
    def get_features(self):
        return self.features
 
 

def run_inference_with_features(model, feature_extractor, dataloader, device, metadata, tau, class_names):

    model.eval()
    
    all_filenames = []
    all_features = []
    all_probs = []
    all_preds = []
    all_confidences = []
    
    print(f"Confidence threshold (tau): {tau}")
    
    with torch.no_grad():
        for images, filenames in tqdm(dataloader, desc="Inference + Features"):
            images = images.to(device)
            
            output = model(images)
            if isinstance(output, (list, tuple)):
                output = output[0]
            
            features = feature_extractor.get_features()
            
            probs = F.softmax(output, dim=1)
            max_probs, preds = probs.max(1)
            
            all_filenames.extend(filenames)
            all_features.append(features.numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_confidences.append(max_probs.cpu().numpy())
    
    all_features = np.vstack(all_features)
    all_probs = np.vstack(all_probs)
    all_preds = np.concatenate(all_preds)
    all_confidences = np.concatenate(all_confidences)
    
    print("\nExtracting metadata for each cutout...")
    target_ids = []
    is_cluster = []
    num_objects = []
    neighbor_ids_list = []
    neighbor_counts = []
    
    for filename in all_filenames:
        if filename in metadata:
            meta = metadata[filename]
            target_ids.append(meta['target_id'])
            is_cluster.append(meta['is_cluster'])
            num_objects.append(meta['num_objects'])
            neighbor_ids_list.append(meta['neighbor_ids'])
            neighbor_counts.append(meta['neighbor_count'])
        else:
            print(f"Warning: No metadata for {filename}")
            target_ids.append(-1)
            is_cluster.append(False)
            num_objects.append(1)
            neighbor_ids_list.append([])
            neighbor_counts.append(0)
    
    results = {
        'filenames': np.array(all_filenames),
        'target_ids': np.array(target_ids),
        'features': all_features,
        'predictions': all_preds,
        'confidences': all_confidences,
        'probs': all_probs,
        'is_cluster': np.array(is_cluster),
        'num_objects': np.array(num_objects),
        'neighbor_ids': np.array(neighbor_ids_list, dtype=object),
        'neighbor_count': np.array(neighbor_counts)
    }
    
    print(f"Total images: {len(results['filenames'])}")
    print(f"Feature dimension: {results['features'].shape[1]}")
    print(f"Solo images: {np.sum(~results['is_cluster'])}")
    print(f"Cluster images: {np.sum(results['is_cluster'])}")
    
    return results
 
 
def organize_images(results, original_dir, output_dir, tau, class_names):

    print("\nOrganizing images into folders...")
    
    os.makedirs(output_dir, exist_ok=True)
    for class_name in class_names:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'unconf_set'), exist_ok=True)
    
    confident_count = 0
    unconfident_count = 0
    class_counts = {name: 0 for name in class_names}
    
    for i in tqdm(range(len(results['filenames'])), desc="Organizing images"):
        filename = results['filenames'][i]
        pred_class = results['predictions'][i]
        confidence = results['confidences'][i]
        is_confident = confidence >= tau
        
        src_path = os.path.join(original_dir, filename)
        
        if not os.path.exists(src_path):
            print(f"\nWarning: Original image not found: {src_path}")
            continue
        
        if is_confident:
            class_name = class_names[pred_class]
            dst_path = os.path.join(output_dir, class_name, filename)
            shutil.copy2(src_path, dst_path)
            confident_count += 1
            class_counts[class_name] += 1
        else:
            dst_path = os.path.join(output_dir, 'unconf_set', filename)
            shutil.copy2(src_path, dst_path)
            unconfident_count += 1
    
    print(f"Total images processed: {len(results['filenames'])}")
    print(f"Confident predictions (>= {tau}): {confident_count}")
    print(f"Unconfident predictions: {unconfident_count}")
    print("\nPer-class breakdown (confident only):")
    for class_name in class_names:
        print(f"  {class_name}: {class_counts[class_name]}")
 
 
def save_results(results, output_dir, features_output, tau, class_names):

    csv_path = os.path.join(output_dir, 'predictions.csv')
    print(f"\nSaving predictions to {csv_path}...")
    
    with open(csv_path, 'w', newline='') as csvfile:
        header = ['filename', 'target_id']
        header.extend([f'{name}_prob' for name in class_names])
        header.extend(['predicted_class', 'predicted_class_name', 'confidence', 
                      'is_confident', 'is_cluster', 'num_objects', 'neighbor_count'])
        
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for i in range(len(results['filenames'])):
            row = [
                results['filenames'][i],
                results['target_ids'][i]
            ]
            row.extend(results['probs'][i].tolist())
            row.extend([
                results['predictions'][i],
                class_names[results['predictions'][i]],
                results['confidences'][i],
                results['confidences'][i] >= tau,
                results['is_cluster'][i],
                results['num_objects'][i],
                results['neighbor_count'][i]
            ])
            writer.writerow(row)
    
    print(f"Saved {len(results['filenames'])} predictions to {csv_path}")
    
    print(f"\nSaving features + metadata to {features_output}...")
    os.makedirs(os.path.dirname(features_output), exist_ok=True)
    
    np.savez_compressed(
        features_output,
        filenames=results['filenames'],
        target_ids=results['target_ids'],
        features=results['features'],
        predictions=results['predictions'],
        confidences=results['confidences'],
        probs=results['probs'],
        is_cluster=results['is_cluster'],
        num_objects=results['num_objects'],
        neighbor_ids=results['neighbor_ids'],
        neighbor_count=results['neighbor_count']
    )
    
    print(f"Saved features to {features_output}")
    print(f"  - filenames: {results['filenames'].shape}")
    print(f"  - features: {results['features'].shape}")
    print(f"  - predictions: {results['predictions'].shape}")
    print(f"  - is_cluster: {results['is_cluster'].shape}")

def main():
    args = parse_args()
    update_config(config, args)
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("="*70)
    logger.info("JADES INFERENCE + FEATURE EXTRACTION")
    logger.info("="*70)
    logger.info(f"Config: {args.cfg}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Masked cutouts: {args.jades_dir}")
    logger.info(f"Original cutouts: {args.original_dir}")
    logger.info(f"Metadata: {args.metadata_file}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Features output: {args.features_output}")
    logger.info(f"Tau: {args.tau}")
    logger.info("="*70)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    setup_cudnn(config)
    
    # Define class names
    class_names = ['class1', 'class2', 'class3', 'class4', 'class5']
    
    # Load metadata
    logger.info(f"\nLoading metadata from {args.metadata_file}...")
    with open(args.metadata_file, 'r') as f:
        metadata = json.load(f)
    logger.info(f"Loaded metadata for {len(metadata)} cutouts")
    
    # Build model
    logger.info("\nBuilding model...")
    model = build_model(config)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}...")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    state = torch.load(args.checkpoint, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state_dict = strip_prefix_if_present(state['state_dict'], 'module.')
    else:
        state_dict = state
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully!")
    
    # Setup feature extractor
    logger.info("Setting up feature extraction hook...")
    feature_extractor = FeatureExtractor(model)
    
    # Build transforms
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = T.Compose([
        T.Resize((args.target_res, args.target_res)),
        T.Resize((224, 224)),
        T.ToTensor(),
        normalize
    ])
    
    # Build dataset and dataloader
    logger.info(f"\nLoading MASKED cutouts from {args.jades_dir}...")
    dataset = JADESInferenceDataset(args.jades_dir, transform=transform)
    logger.info(f"Found {len(dataset)} masked images")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Run inference + feature extraction
    results = run_inference_with_features(
        model, feature_extractor, dataloader, device, metadata, args.tau, class_names
    )
    
    # Organize ORIGINAL images into folders
    organize_images(results, args.original_dir, args.output_dir, args.tau, class_names)
    
    # Save results (CSV + NPZ)
    save_results(results, args.output_dir, args.features_output, args.tau, class_names)
    
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("  - Images organized into class folders (ORIGINAL images)")
    logger.info("  - predictions.csv (with cluster metadata)")
    logger.info(f"  - {args.features_output} (features + metadata for visualization)")
 
 
if __name__ == "__main__":
    main()