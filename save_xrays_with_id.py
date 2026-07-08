"""
Generate and save X-rays from CT volumes with ID-based naming.

This script:
1. Loads CT volumes from the dataset
2. Generates X-rays using DRR (Digitally Reconstructed Radiographs)
3. Saves X-rays as PNG with ID_xray1.png, ID_xray2.png naming
4. All images are saved in [0, 255] range
"""

import argparse
import os
import sys
from typing import List

import cv2
import numpy as np
import torch
import torch as th
from tqdm import tqdm
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conds.ct2xrays import get_xrays_from_ct
from conds.utils import create_val_loader
from util import load_config, normalize_image


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and save X-rays from CT volumes with ID-based naming"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="lidc.yaml",
        help="Config file name (in configs/ directory)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="xrays_output",
        help="Output directory to save X-ray PNG files"
    )
    parser.add_argument(
        "--rotations",
        type=int,
        default=2,
        help="Number of X-ray projections to generate (default: 2)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation (cuda or cpu)"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="val",
        help="Data type to process (val or test)"
    )
    return parser.parse_args()


def sanitize_id(sample_id: str) -> str:
    """Sanitize sample ID for filesystem use by extracting core identifier.
    
    Extracts the core identifier from paths like:
    - ../../data/LIDC-HDF5-256/LIDC-IDRI-0256.20000101.8658.4.1/ct_xray_data.h5
      -> LIDC-IDRI-0256.20000101.8658.4.1
    
    Args:
        sample_id: Sample ID from dataset (can be a path or filename)
    
    Returns:
        Sanitized identifier safe for filesystem use
    """
    # Check if the path contains a directory with LIDC-IDRI pattern
    path_parts = sample_id.split(os.sep)
    for part in reversed(path_parts):
        if part.startswith('LIDC-IDRI-'):
            return part
    
    # Fallback to basename processing
    basename = os.path.basename(sample_id)
    
    # Remove common file extensions
    for ext in ['.h5', '.nii.gz', '.nii', '.hdf5']:
        if basename.endswith(ext):
            basename = basename[:-len(ext)]
            break
    
    # Remove common suffixes
    common_suffixes = [
        '_ct_xray_data',
        '_ct_128_norm',
        '_ct_256_norm',
        '_ct_64_norm',
        '_ct_norm',
        '_norm',
        '_ct',
    ]
    
    common_suffixes.sort(key=len, reverse=True)
    for suffix in common_suffixes:
        if basename.endswith(suffix):
            basename = basename[:-len(suffix)]
            break
    
    sanitized = basename.strip(". _")
    
    if not sanitized:
        sanitized = basename.replace(os.sep, "_").replace("/", "_").replace("\\", "_")
        sanitized = sanitized.strip(". _")
    
    return sanitized


def save_xrays_as_png(
    xrays: List[th.Tensor],
    output_dir: str,
    sample_id: str,
    logger: logging.Logger
) -> None:
    """Save generated X-rays as PNG images with ID-based naming.
    
    Args:
        xrays: List of X-ray tensors generated from CT
        output_dir: Directory to save the X-rays
        sample_id: Sample ID for naming
        logger: Logger instance
    """
    os.makedirs(output_dir, exist_ok=True)
    
    sanitized_id = sanitize_id(sample_id)
    
    logger.debug(f"  Saving {len(xrays)} X-rays for {sanitized_id}")
    
    for i, xray in enumerate(xrays):
        # Convert to numpy
        xray_np = th.as_tensor(xray.data).cpu().squeeze().numpy()
        
        # Normalize to [0, 255] range
        xray_np = normalize_image(xray_np)
        xray_np = (xray_np * 255).astype(np.uint8)
        
        # Save with ID_xray{i+1}.png format
        filename = f"{sanitized_id}_xray{i+1}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, xray_np)


def process_sample(
    idx: int,
    dataset,
    args: argparse.Namespace,
    device: th.device,
    logger: logging.Logger
) -> None:
    """Process a single sample: generate X-rays from CT and save as PNG.
    
    Args:
        idx: Sample index
        dataset: Dataset instance
        args: Command line arguments
        device: Device to use
        logger: Logger instance
    """
    # Get sample data
    sample_data = dataset[idx]
    datarow = sample_data["image"]
    sample_id = sample_data.get("id", f"sample_{idx}")
    
    sanitized_id = sanitize_id(sample_id)
    
    logger.info(f"[{idx+1}] Processing: {sanitized_id}")
    
    # Generate X-rays from CT using DRR
    xrays = get_xrays_from_ct(
        datarow,
        idx=idx,
        device=device,
        rotations=args.rotations
    )
    
    # Save X-rays directly to output directory (no subdirectories)
    save_xrays_as_png(xrays, args.output_dir, sample_id, logger)
    logger.info(f"  ✓ Saved {len(xrays)} X-rays for: {sanitized_id}")


def main():
    """Main function."""
    args = parse_args()
    logger = setup_logging()
    
    logger.info("="*80)
    logger.info("Generate and Save X-rays from CT Volumes with ID-based Naming")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  Config: {args.config}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Rotations: {args.rotations}")
    logger.info(f"  Device: {args.device}")
    logger.info("="*80)
    
    # Load configuration
    config = load_config(os.path.join("configs", args.config))
    
    # Setup device
    device = th.device(args.device)
    th.manual_seed(config.training.seed)
    
    # Load dataset
    logger.info("\nLoading dataset...")
    dataset = create_val_loader(config, data_type=args.data_type)
    logger.info(f"Loaded dataset with {len(dataset)} samples")
    
    # Determine number of samples to process
    num_samples = args.num_samples if args.num_samples else len(dataset)
    num_samples = min(num_samples, len(dataset))
    logger.info(f"Processing {num_samples} samples\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process samples
    success_count = 0
    failed_samples = []
    
    for idx in tqdm(range(num_samples), desc="Generating X-rays"):
        try:
            process_sample(idx, dataset, args, device, logger)
            success_count += 1
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            failed_samples.append((idx, str(e)))
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("Processing Complete!")
    logger.info("="*80)
    logger.info(f"Total samples: {num_samples}")
    logger.info(f"Successfully saved: {success_count}")
    logger.info(f"Failed: {len(failed_samples)}")
    
    if failed_samples:
        logger.info("\nFailed samples:")
        for idx, error in failed_samples:
            logger.info(f"  Sample {idx}: {error}")
    
    logger.info(f"\nAll outputs saved to: {args.output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
