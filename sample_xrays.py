"""X-ray conditional sampling from pre-trained DiT models."""
import argparse
from contextlib import contextmanager, redirect_stdout
import json
import logging
import os
import time
import warnings
from typing import Any, Dict, List, Optional

import cv2
import nibabel as nib
import numpy as np
import torch
import torch as th
from tqdm import tqdm

from conds.ct2xrays import get_xrays_from_ct
from conds.utils import (
    compute_volume_metrics,
    create_val_loader,
    gen_rotation_translate,
    initialize_paths,
)
from datasets.lidc import data_transform_backward
from diffusion.image_noise_diffusion import XrayGuidedIaNDiffusion
from utils.download import find_model
from util import (
    evaluation_metrics,
    load_config,
    normalize_image,
    Peak_Signal_to_Noise_Rate_3D,
    remove_empty_directories,
    save_evaluation_samples,
    Structural_Similarity,
    tensor_back_to_unMinMax,
)
from models import DiT_models

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEBUG = True


@contextmanager
def quiet_stdout(enabled: bool):
    """Suppress detailed diagnostics while preserving stderr progress bars."""
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as sink, redirect_stdout(sink):
        yield


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="X-ray conditional sampling")
    parser.add_argument("--config", type=str, default="lidc_stage2_global.yaml", help="Config file path")
    parser.add_argument("--num-samples", type=int, default=102, help="Number of samples")
    parser.add_argument("--ckpt", type=str, default="results/001-DiT-B-12-12/checkpoints/013000.pt", help="Model checkpoint path")
    parser.add_argument("--num-sampling-steps", type=int, default=1000, help="Sampling steps")
    parser.add_argument("--output-dir", type=str, default="outputs_Cond", help="Output directory")
    parser.add_argument("--new", action="store_true", help="Use new sampling schema")
    parser.add_argument("--rotations", type=int, default=2, help="Number of rotations")
    parser.add_argument("--num-save-samples", type=int, default=102, help="Number of samples to save to disk (to limit log size)")
    parser.add_argument("--no-save-intermediate", action="store_true", help="Disable saving intermediate images and volumes (only compute metrics)")
    parser.add_argument("--save-nifti", action="store_true", help="Save NIfTI (.nii.gz) files (disabled by default when --no-save-intermediate is set)")
    parser.add_argument("--save-png", action="store_true", help="Save PNG images (disabled by default when --no-save-intermediate is set)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show model, dataset, DRR, and per-step diagnostic logs")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    return args


def setup_device(config: Dict[str, Any]) -> th.device:
    """Setup device and random seed."""
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    th.manual_seed(config.training.seed)
    return device


def load_model(config: Dict[str, Any], args: argparse.Namespace, device: th.device, logger: logging.Logger) -> th.nn.Module:
    """Load diffusion model from checkpoint."""
    model = DiT_models[config.model.name](
        input_size=config.data.image_size,
        in_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        learn_sigma=True if config.model.out_channels == 2 else False,
        flash_attn=config.model.flash_attn
    ).to(device)

    state_dict = find_model(args.ckpt)
    model.load_state_dict(state_dict)
    logger.info(f"Loaded model from {args.ckpt}")
    model.eval()
    logger.info("Model ready for inference")
    return model


def setup_diffusion(args: argparse.Namespace) -> XrayGuidedIaNDiffusion:
    """Setup diffusion process with specified sampling steps."""
    diffusion = XrayGuidedIaNDiffusion(timestep_respacing=str(args.num_sampling_steps), loss_type="l2")
    diffusion.config.debug = args.verbose
    diffusion.config.save_detailed = args.verbose
    return diffusion


def setup_data(config: Dict[str, Any], device: th.device, logger: logging.Logger, debug_dir: str) -> Any:
    """Load validation dataset."""
    dataset = create_val_loader(config)
    logger.info(f"Loaded dataset with {len(dataset)} samples")
    return dataset


def generate_diffusion_samples(
    model: th.nn.Module,
    diffusion: XrayGuidedIaNDiffusion,
    conditions: List[th.Tensor],
    idx: int,
    datarow: th.Tensor,
    args: argparse.Namespace,
    config: Dict[str, Any],
    device: th.device,
    sample_dir: Optional[str]
) -> th.Tensor:
    """Run diffusion sampling conditioned on X-ray projections."""
    z = th.randn(
        1, 
        config.model.in_channels, 
        config.data.image_size, 
        config.data.image_size, 
        config.data.image_size, 
        device=device
    )

    samples = diffusion.p_sample_loop(
        model=model.forward, 
        z=z, 
        conditions=conditions,
        idx=idx,
        ref_vol=datarow[0].to(device),
        device=device,
        new_sampling=args.new,
        rotations=args.rotations,
        sample_dir=sample_dir,
        model_kwargs={"y": None} if config.model.num_classes else {}
    )
    
    return samples


def generate_conditional_samples(
    dataset: Any,
    model: th.nn.Module,
    diffusion: XrayGuidedIaNDiffusion,
    args: argparse.Namespace,
    config: Dict[str, Any],
    device: th.device,
    logger: logging.Logger,
    output_base: str,
    samples_dir: str,
    xray_verification_dir: str
) -> None:
    """Generate conditional samples using X-ray conditions and diffusion."""
    generation_times = []
    condition_times = []
    volume_metrics = []
    
    select_idxs = range(len(dataset))[:args.num_samples]
    num_save_samples = min(args.num_save_samples, len(select_idxs))
    
    # Determine save flags based on args
    if args.no_save_intermediate:
        save_intermediate = False
        save_nifti = args.save_nifti
        save_png = args.save_png
        logger.info(f"Processing {len(select_idxs)} samples, computing metrics ONLY (no intermediate files will be saved)")
    else:
        save_intermediate = True
        save_nifti = True
        save_png = True
        logger.info(f"Processing {len(select_idxs)} samples, saving intermediate results for first {num_save_samples} samples, metrics for all samples")
    
    for idx in tqdm(select_idxs, desc="Generating samples"):
        start_time = time.time()
        datarow = dataset[idx]["image"]
        x_rays = dataset[idx]["x_rays"]
        sample_id = dataset[idx].get("id", f"sample_{idx}")  # Extract id, fallback to idx if not available
        if args.verbose:
            print(f"Sample ID: {sample_id}")
        sanitized_id = sanitize_id(sample_id) if sample_id else f"sample_{idx}"
        if args.verbose:
            print(f"Sanitized ID: {sanitized_id}")
        # exit(0)
        if idx == 0:  # Log sample ID format for first sample
            logger.info(f"Sample ID format: original='{sample_id}' -> sanitized='{sanitized_id}'")
        
        cond_start = time.time()
        with quiet_stdout(not args.verbose):
            conditions = get_xrays_from_ct(
                datarow, idx=idx,
                device=device,
                rotations=args.rotations
            )
        condition_times.append(time.time() - cond_start)
        
        should_save_intermediate = save_intermediate and (idx < num_save_samples)
        
        # Create sample directory under samples_dir using sample_id
        sample_dir = create_sample_directories(samples_dir, sample_id, idx) if should_save_intermediate else None
        
        if should_save_intermediate:
            if DEBUG:
                # Use sanitized sample_id for directory name in xray_verification_dir
                sample_xray_dir = os.path.join(xray_verification_dir, sanitized_id)
                os.makedirs(sample_xray_dir, exist_ok=True)
                with quiet_stdout(not args.verbose):
                    save_xray_verification(conditions, sample_xray_dir, idx, save_png=save_png)
                    save_reference_xrays(x_rays, sample_xray_dir, idx, save_png=save_png)
        
        with quiet_stdout(not args.verbose):
            samples = generate_diffusion_samples(
                model, diffusion, conditions,
                idx, datarow, args, config, device, sample_dir
            )
        
        if should_save_intermediate:
            with quiet_stdout(not args.verbose):
                save_reference_data(datarow, conditions, sample_dir, save_nifti=save_nifti, save_png=save_png)
                save_generated_samples(samples, sample_dir, config, logger, save_nifti=save_nifti, save_png=save_png)
        
        # print("="*30)
        # print(f"Output samples shape: {samples.shape}, value range: {samples.min():.6f}, {samples.max():.6f}, {samples.mean():.6f}, {samples.std():.6f}")
        samples = data_transform_backward(samples)
        # print(f"After backward transform: {samples.min():.6f}, {samples.max():.6f}, {samples.mean():.6f}, {samples.std():.6f}")
        samples = th.clamp(samples, 0.0, 1.0)
        # print(f"After clamp: {samples.min():.6f}, {samples.max():.6f}, {samples.mean():.6f}, {samples.std():.6f}")
        # print("="*30)
        
        datarow = data_transform_backward(datarow)
        
        metrics = evaluation_metrics(datarow, samples, verbose=args.verbose)
        volume_metrics.append(metrics)
        
        generation_time = time.time() - start_time
        generation_times.append(generation_time)
        
        if sample_dir:
            save_sample_metrics(metrics, sample_dir, idx, generation_time, condition_times[-1] if condition_times else 0)
            remove_empty_directories(sample_dir)
        
        log_sample_progress(
            idx, select_idxs, generation_times, condition_times, 
            metrics, logger
        )
    
    with quiet_stdout(not args.verbose):
        log_final_statistics(select_idxs, generation_times,
                             condition_times, volume_metrics,
                             output_base, logger)
    remove_empty_directories(output_base)


def generate_xray_verification(
    dataset: Any,
    args: argparse.Namespace,
    device: th.device,
    logger: logging.Logger,
    xray_verification_dir: str
) -> None:
    """Generate X-ray verification samples from dataset."""
    logger.info("Starting X-ray verification for all evaluation samples...")
    
    select_idxs = range(len(dataset))[:args.num_samples]
    
    for idx in tqdm(select_idxs, desc="Generating X-rays for verification"):
        datarow = dataset[idx]["image"]
        sample_id = dataset[idx].get("id", f"sample_{idx}")  # Extract id, fallback to idx if not available
        sanitized_id = sanitize_id(sample_id) if sample_id else f"sample_{idx}"
        sample_xray_dir = os.path.join(xray_verification_dir, sanitized_id)
        os.makedirs(sample_xray_dir, exist_ok=True)
        
        xrays = get_xrays_from_ct(
            datarow, idx=idx,
            device=device
        )
        
        save_xray_verification(xrays, sample_xray_dir, idx)


def sanitize_id(sample_id: str) -> str:
    """Sanitize sample ID for filesystem use by extracting core identifier.
    
    Extracts the core identifier from paths like:
    - data/LIDC-HDF5-256/LIDC-IDRI-0256.20000101.8658.4.1/ct_xray_data.h5
      -> LIDC-IDRI-0256.20000101.8658.4.1
    - data/LIDC-HDF5-256/LIDC-IDRI-0256.20000101.8658.4.1_ct_xray_data.h5
      -> LIDC-IDRI-0256.20000101.8658.4.1
    
    Args:
        sample_id: Sample ID from dataset (can be a path or filename)
    
    Returns:
        Sanitized identifier safe for filesystem use
    """
    # Check if the path contains a directory with LIDC-IDRI pattern
    # For paths like: ../../data/LIDC-HDF5-256/LIDC-IDRI-0256.20000101.8658.4.1/ct_xray_data.h5
    path_parts = sample_id.split(os.sep)
    for part in reversed(path_parts):
        if part.startswith('LIDC-IDRI-'):
            return part
    
    # Fallback to basename processing for other formats
    basename = os.path.basename(sample_id)
    
    # Remove common file extensions
    for ext in ['.h5', '.nii.gz', '.nii', '.hdf5']:
        if basename.endswith(ext):
            basename = basename[:-len(ext)]
            break
    
    # Remove common suffixes that are not part of the core identifier
    # Patterns to remove: _ct_xray_data, _ct_128_norm, _norm, _ct, etc.
    common_suffixes = [
        '_ct_xray_data',
        '_ct_128_norm',
        '_ct_256_norm',
        '_ct_64_norm',
        '_ct_norm',
        '_norm',
        '_ct',
    ]
    
    # Remove suffixes in order (longest first to avoid partial matches)
    common_suffixes.sort(key=len, reverse=True)
    for suffix in common_suffixes:
        if basename.endswith(suffix):
            basename = basename[:-len(suffix)]
            break
    
    # Clean up: remove leading/trailing dots, spaces, and underscores
    sanitized = basename.strip(". _")
    
    # If we ended up with an empty string, fall back to original basename processing
    if not sanitized:
        # Fallback: use basename and replace invalid characters
        sanitized = basename.replace(os.sep, "_").replace("/", "_").replace("\\", "_")
        sanitized = sanitized.strip(". _")
    
    return sanitized


def create_sample_directories(output_base: str, sample_id: str, idx: int = None) -> str:
    """Create sample directories for reference and generated outputs.
    
    Args:
        output_base: Base output directory
        sample_id: Sample ID from dataset (will be sanitized for filesystem)
        idx: Optional index for fallback if sample_id is empty
    """
    # Sanitize the sample_id for filesystem use
    sanitized_id = sanitize_id(sample_id) if sample_id else f"sample_{idx}"
    
    # Use sanitized_id, fallback to idx if sanitized_id is empty
    if not sanitized_id and idx is not None:
        sanitized_id = f"sample_{idx}"
    elif not sanitized_id:
        sanitized_id = "sample_unknown"
    
    sample_dir = os.path.join(output_base, sanitized_id)
    reference_dir = os.path.join(sample_dir, "reference")
    generated_dir = os.path.join(sample_dir, "generated")
    
    for d in [sample_dir, reference_dir, generated_dir]:
        os.makedirs(d, exist_ok=True)
    
    return sample_dir


def print_xray_statistics(xrays_np: np.ndarray, xray_type: str = "X-rays") -> None:
    """Print statistics for X-ray data.
    
    Args:
        xrays_np: X-ray data as numpy array
        xray_type: Type label for the X-rays (e.g., "Estimated X-rays", "Reference X-rays")
    """
    print(f"\n{xray_type} Statistics:")
    print(f"  Shape: {xrays_np.shape}")
    print(f"  Min: {xrays_np.min():.6f}")
    print(f"  Max: {xrays_np.max():.6f}")
    print(f"  Mean: {xrays_np.mean():.6f}")
    print(f"  Std: {xrays_np.std():.6f}")
    
    if xrays_np.ndim >= 2:
        print(f"  Data type: {xrays_np.dtype}")
        if xrays_np.ndim == 3:
            for i in range(xrays_np.shape[0]):
                view = xrays_np[i]
                print(f"  View {i}: min={view.min():.6f}, max={view.max():.6f}, mean={view.mean():.6f}, std={view.std():.6f}")


def save_xray_verification(xrays: List[th.Tensor], sample_xray_dir: str, idx: int, save_png: bool = True) -> None:
    """Save X-ray verification data (numpy arrays and PNG images)."""
    if not save_png:
        return
    
    os.makedirs(sample_xray_dir, exist_ok=True)
    
    xrays_tensor = th.stack([th.as_tensor(x.data) for x in xrays]).cpu()
    xrays_np = xrays_tensor.numpy()
    
    # print_xray_statistics(xrays_np, "Estimated X-rays")
    
    np.save(os.path.join(sample_xray_dir, "xrays.npy"), xrays_np)
    print(f"Saved xrays with shape {xrays_np.shape} to {os.path.join(sample_xray_dir, 'xrays.npy')}")
    
    xray_images = []
    for i, xray in enumerate(xrays):
        xray_np = th.as_tensor(xray.data).cpu().squeeze().numpy()
        xray_np = normalize_image(xray_np)
        xray_np = (xray_np * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(sample_xray_dir, f"xray_{i:03d}.png"), xray_np)
        xray_images.append(xray_np)
    
    combined = np.hstack(xray_images)
    cv2.imwrite(os.path.join(sample_xray_dir, "combined_xrays.png"), combined)


def save_reference_xrays(xrays: Optional[th.Tensor], sample_xray_dir: str, idx: int, target_size: int = 128, save_png: bool = True) -> None:
    """Save reference X-rays from dataset (numpy arrays and PNG images).
    
    Args:
        xrays: Reference X-rays tensor from dataset
        sample_xray_dir: Directory to save the X-rays
        idx: Sample index
        target_size: Target size for resizing (default: 128)
        save_png: Whether to save PNG files (default: True)
    """
    if not save_png or xrays is None:
        if xrays is None:
            print(f"No reference X-rays available for sample {idx}")
        return
    
    os.makedirs(sample_xray_dir, exist_ok=True)
    
    if isinstance(xrays, th.Tensor):
        xrays_np = xrays.cpu().numpy()
    else:
        xrays_np = np.asarray(xrays)
    
    if xrays_np.ndim == 4 and xrays_np.shape[0] == 1:
        xrays_np = xrays_np.squeeze(0)
    
    original_shape = xrays_np.shape
    print(f"Original reference xrays shape: {original_shape}")
    
    if xrays_np.ndim == 3:
        resized_xrays = []
        for i in range(xrays_np.shape[0]):
            xray = xrays_np[i]
            if xray.shape[0] != target_size or xray.shape[1] != target_size:
                xray_resized = cv2.resize(xray, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            else:
                xray_resized = xray
            resized_xrays.append(xray_resized)
        xrays_np = np.stack(resized_xrays, axis=0)
    elif xrays_np.ndim == 2:
        if xrays_np.shape[0] != target_size or xrays_np.shape[1] != target_size:
            xrays_np = cv2.resize(xrays_np, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    print(f"Resized reference xrays shape: {xrays_np.shape}")
    
    # print_xray_statistics(xrays_np, "Reference X-rays")
    
    np.save(os.path.join(sample_xray_dir, "reference_xrays.npy"), xrays_np)
    print(f"Saved reference xrays with shape {xrays_np.shape} to {os.path.join(sample_xray_dir, 'reference_xrays.npy')}")
    
    num_views = xrays_np.shape[0] if xrays_np.ndim == 3 else 1
    xray_images = []
    
    if xrays_np.ndim == 3:
        for i in range(num_views):
            xray_np = xrays_np[i]
            xray_np = normalize_image(xray_np)
            xray_np = (xray_np * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(sample_xray_dir, f"reference_xray_{i:03d}.png"), xray_np)
            xray_images.append(xray_np)
    elif xrays_np.ndim == 2:
        xray_np = normalize_image(xrays_np)
        xray_np = (xray_np * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(sample_xray_dir, "reference_xray_000.png"), xray_np)
        xray_images.append(xray_np)
    
    if len(xray_images) > 1:
        combined = np.hstack(xray_images)
        cv2.imwrite(os.path.join(sample_xray_dir, "combined_reference_xrays.png"), combined)


def save_reference_data(datarow: th.Tensor, conditions: List[th.Tensor], sample_dir: str, save_nifti: bool = True, save_png: bool = True) -> None:
    """Save reference CT and X-ray projection data."""
    if not save_nifti and not save_png:
        return
    
    reference_dir = os.path.join(sample_dir, "reference")
    os.makedirs(reference_dir, exist_ok=True)
    
    if save_nifti:
        img = nib.Nifti1Image(datarow[0].numpy(), np.eye(4))
        nib.save(img, os.path.join(reference_dir, f"original_ct.nii.gz"))
    
    # Save middle slices of reference CT volume as individual PNGs
    save_middle_slices_as_png(datarow, reference_dir, enabled=save_png)
    
    # Save conditions/projections as individual PNG files
    if save_png:
        saved_images = []
        for i, proj in enumerate(conditions):
            proj_np = proj.cpu().numpy()
            if proj_np.ndim > 2:
                proj_np = proj_np[0] if proj_np.shape[0] == 1 else proj_np.squeeze()
            proj_np = normalize_image(proj_np)
            proj_np = (proj_np * 255).astype(np.uint8)
            filename = os.path.join(reference_dir, f"projection_{i:03d}.png")
            cv2.imwrite(filename, proj_np)
            saved_images.append(proj_np)
        
        # Create combined image only if we have multiple projections with compatible shapes
        if len(saved_images) > 1:
            try:
                # Ensure all images have the same height for horizontal stacking
                max_height = max(img.shape[0] for img in saved_images)
                resized_images = []
                for img in saved_images:
                    if img.shape[0] != max_height:
                        # Resize to match height while preserving aspect ratio
                        aspect = img.shape[1] / img.shape[0]
                        new_width = int(max_height * aspect)
                        img = cv2.resize(img, (new_width, max_height), interpolation=cv2.INTER_LINEAR)
                    resized_images.append(img)
                
                combined = np.hstack(resized_images)
                cv2.imwrite(os.path.join(reference_dir, "combined_reference.png"), combined)
            except Exception as e:
                print(f"[WARNING] Could not create combined reference image: {e}")
        
        save_projection_metadata(datarow, conditions, reference_dir)


def save_projection_metadata(datarow: th.Tensor, conditions: List[th.Tensor], reference_dir: str) -> None:
    """Save projection metadata to text file."""
    with open(os.path.join(reference_dir, "projection_info.txt"), "w") as f:
        f.write("Reference X-ray Projections:\n")
        f.write("1. Axial view (top-down)\n")
        f.write("2. Coronal view (front)\n")
        f.write("3. Sagittal view (side)\n")
        f.write(f"\nOriginal CT shape: {datarow.shape}\n")
        f.write(f"Projection shapes: {[p.shape for p in conditions]}\n")


def save_middle_slices_as_png(volume: th.Tensor, save_dir: str, enabled: bool = True) -> None:
    """Save the middle slices (axial, coronal, sagittal) of a 3D volume as individual PNG files.
    
    Args:
        volume: 3D volume tensor with shape [B, C, D, H, W] or [C, D, H, W]
        save_dir: Directory to save the PNG files
        enabled: Whether to actually save the files (default: True)
    """
    if not enabled:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy and handle batch dimension
    if isinstance(volume, th.Tensor):
        vol_np = volume.cpu().numpy()
    else:
        vol_np = np.asarray(volume)
    
    # Remove batch dimension if present
    if vol_np.ndim == 5:
        vol_np = vol_np[0]  # [C, D, H, W]
    
    # Remove channel dimension if present
    if vol_np.ndim == 4:
        vol_np = vol_np[0]  # [D, H, W]
    
    # Get middle indices
    depth, height, width = vol_np.shape
    mid_d = depth // 2
    mid_h = height // 2
    mid_w = width // 2
    
    # Extract middle slices
    axial_slice = vol_np[mid_d, :, :]       # Middle slice along depth (axial view)
    coronal_slice = vol_np[:, mid_h, :]     # Middle slice along height (coronal view)
    sagittal_slice = vol_np[:, :, mid_w]    # Middle slice along width (sagittal view)
    
    # Normalize and save each slice
    slices = {
        'axial': axial_slice,
        'coronal': coronal_slice,
        'sagittal': sagittal_slice
    }
    
    for name, slice_data in slices.items():
        # Normalize to [0, 255]
        slice_min, slice_max = slice_data.min(), slice_data.max()
        if slice_max > slice_min:
            slice_norm = (slice_data - slice_min) / (slice_max - slice_min)
        else:
            slice_norm = np.zeros_like(slice_data)
        
        slice_uint8 = (slice_norm * 255).astype(np.uint8)
        
        # Save as PNG
        filename = os.path.join(save_dir, f"middle_slice_{name}.png")
        cv2.imwrite(filename, slice_uint8)
    
    print(f"[+] Saved middle slices to: {save_dir}")


def save_generated_samples(samples: th.Tensor, sample_dir: str, config: Dict[str, Any], logger: logging.Logger, save_nifti: bool = True, save_png: bool = True) -> None:
    """Save generated samples to directory with proper backward transformation."""
    if not save_nifti and not save_png:
        return
    
    generated_dir = os.path.join(sample_dir, "generated")
    os.makedirs(generated_dir, exist_ok=True)
    
    samples_normalized = data_transform_backward(samples)
    samples_normalized = th.clamp(samples_normalized, 0.0, 1.0)
    
    if save_nifti:
        save_evaluation_samples(
            samples_normalized, 
            generated_dir, 
            config.data.image_size,
            epoch=0, 
            logger=logger
        )
    
    # Save middle slices as individual PNGs
    save_middle_slices_as_png(samples_normalized, generated_dir, enabled=save_png)


def save_sample_metrics(metrics: Dict[str, float], sample_dir: str, idx: int, generation_time: float, condition_time: float) -> None:
    """Save evaluation metrics for a single sample to files.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        sample_dir: Directory to save the metrics file
        idx: Sample index
        generation_time: Time taken for generation (seconds)
        condition_time: Time taken for condition generation (seconds)
    """
    metrics_file = os.path.join(sample_dir, "metrics.txt")
    metrics_json = os.path.join(sample_dir, "metrics.json")
    
    with open(metrics_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Evaluation Metrics for Sample {idx}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Performance Metrics:\n")
        f.write(f"  Generation Time: {generation_time:.4f} s\n")
        f.write(f"  Condition Time: {condition_time:.4f} s\n")
        f.write(f"  Total Time: {generation_time + condition_time:.4f} s\n\n")
        f.write("Volume Comparison Metrics:\n")
        f.write(f"  MSE: {metrics['mse']:.8f}\n")
        f.write(f"  PSNR: {metrics['psnr']:.8f} dB\n")
        f.write(f"  SSIM: {metrics['ssim']:.8f}\n")
        if 'snr' in metrics:
            f.write(f"  SNR: {metrics['snr']:.8f} dB\n")
            f.write(f"  Signal Power: {metrics.get('signal_power', 0):.8f}\n")
            f.write(f"  Noise Power: {metrics.get('noise_power', 0):.8f}\n")
        f.write("\n" + "=" * 60 + "\n")
    
    metrics_with_meta = {
        "sample_index": int(idx),
        "generation_time": float(generation_time),
        "condition_time": float(condition_time),
        "total_time": float(generation_time + condition_time),
        **{k: float(v) for k, v in metrics.items()}
    }
    
    with open(metrics_json, "w") as f:
        json.dump(metrics_with_meta, f, indent=2)


def compute_and_save_metrics(datarow: th.Tensor, samples: th.Tensor, sample_dir: str, volume_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute MSE, PSNR, SSIM, SNR and save to file."""
    reference_volume = datarow[0].numpy()
    generated_volume = samples[0, 0].float().cpu().numpy()
    metrics = compute_volume_metrics(reference_volume, generated_volume)
    volume_metrics.append(metrics)
    
    metrics_file = os.path.join(sample_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Volume Comparison Metrics:\n")
        f.write(f"MSE: {metrics['mse']:.6f}\n")
        f.write(f"PSNR: {metrics['psnr']:.2f} dB\n")
        f.write(f"SSIM: {metrics['ssim']:.4f}\n")
        if 'snr' in metrics:
            f.write(f"SNR: {metrics['snr']:.2f} dB\n")
            f.write(f"Signal Power: {metrics.get('signal_power', 0):.6f}\n")
            f.write(f"Noise Power: {metrics.get('noise_power', 0):.6f}\n")
    
    return metrics

def save_final_metrics(avg_metrics: Dict[str, float], std_metrics: Dict[str, float], generation_times: List[float], condition_times: List[float], num_samples: int, output_base: str) -> None:
    """Save final average evaluation metrics to files.
    
    Args:
        avg_metrics: Dictionary containing average metrics
        std_metrics: Dictionary containing standard deviation of metrics
        generation_times: List of generation times for each sample
        condition_times: List of condition generation times for each sample
        num_samples: Total number of samples
        output_base: Base output directory
    """
    final_metrics_file = os.path.join(output_base, "final_metrics.txt")
    final_metrics_json = os.path.join(output_base, "final_metrics.json")
    
    total_time = sum(generation_times)
    avg_gen_time = np.mean(generation_times)
    avg_cond_time = np.mean(condition_times)
    
    with open(final_metrics_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Final Evaluation Results Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Samples: {num_samples}\n\n")
        f.write("Performance Statistics:\n")
        f.write(f"  Average Generation Time: {avg_gen_time:.4f} s\n")
        f.write(f"  Average Condition Time: {avg_cond_time:.4f} s\n")
        f.write(f"  Total Time: {total_time:.4f} s ({total_time/60:.4f} min)\n\n")
        f.write("Average Volume Metrics (Mean ± Std):\n")
        f.write(f"  MSE: {avg_metrics['mse']:.8f} ± {std_metrics['mse']:.8f}\n")
        f.write(f"  PSNR: {avg_metrics['psnr']:.8f} ± {std_metrics['psnr']:.8f} dB\n")
        f.write(f"  SSIM: {avg_metrics['ssim']:.8f} ± {std_metrics['ssim']:.8f}\n")
        if 'snr' in avg_metrics:
            f.write(f"  SNR: {avg_metrics['snr']:.8f} ± {std_metrics['snr']:.8f} dB\n")
            f.write(f"  Signal Power: {avg_metrics['signal_power']:.8f} ± {std_metrics['signal_power']:.8f}\n")
            f.write(f"  Noise Power: {avg_metrics['noise_power']:.8f} ± {std_metrics['noise_power']:.8f}\n")
        f.write("\n" + "=" * 60 + "\n")
    
    final_metrics_dict = {
        "num_samples": int(num_samples),
        "performance": {
            "avg_generation_time": float(avg_gen_time),
            "avg_condition_time": float(avg_cond_time),
            "total_time": float(total_time),
            "total_time_minutes": float(total_time / 60)
        },
        "metrics": {
            "mean": {k: float(v) for k, v in avg_metrics.items()},
            "std": {k: float(v) for k, v in std_metrics.items()}
        }
    }
    
    with open(final_metrics_json, "w") as f:
        json.dump(final_metrics_dict, f, indent=2)
    
    print(f"\nFinal metrics saved to:")
    print(f"  - {final_metrics_file}")
    print(f"  - {final_metrics_json}")


def log_sample_progress(idx: int, select_idxs: range, generation_times: List[float], condition_times: List[float], metrics: Dict[str, float], logger: logging.Logger) -> None:
    """Log progress for current sample with ETA."""
    avg_time = np.mean(generation_times)
    eta = avg_time * (len(select_idxs) - idx - 1)
    
    logger.info(f"Completed sample {idx+1}/{len(select_idxs)}")
    logger.info(f"  Generation time: {generation_times[-1]:.2f}s")
    logger.info(f"  Condition time: {condition_times[-1]:.2f}s")
    logger.info(f"  Volume metrics:")
    logger.info(f"    - MSE: {metrics['mse']:.6f}")
    logger.info(f"    - PSNR: {metrics['psnr']:.2f} dB")
    logger.info(f"    - SSIM: {metrics['ssim']:.4f}")
    if 'snr' in metrics:
        logger.info(f"    - SNR: {metrics['snr']:.2f} dB")
        logger.info(f"    - Signal Power: {metrics.get('signal_power', 0):.6f}")
        logger.info(f"    - Noise Power: {metrics.get('noise_power', 0):.6f}")
    logger.info(f"  ETA: {eta/60:.2f}min")


def log_final_statistics(select_idxs: range, generation_times: List[float], condition_times: List[float], volume_metrics: List[Dict[str, float]], output_base: str, logger: logging.Logger) -> None:
    """Log final generation statistics and average metrics."""
    avg_metrics = {
        'mse': np.mean([m['mse'] for m in volume_metrics]),
        'psnr': np.mean([m['psnr'] for m in volume_metrics]),
        'ssim': np.mean([m['ssim'] for m in volume_metrics])
    }
    
    if volume_metrics and 'snr' in volume_metrics[0]:
        avg_metrics['snr'] = np.mean([m.get('snr', 0) for m in volume_metrics])
        avg_metrics['signal_power'] = np.mean([m.get('signal_power', 0) for m in volume_metrics])
        avg_metrics['noise_power'] = np.mean([m.get('noise_power', 0) for m in volume_metrics])
    
    std_metrics = {
        'mse': np.std([m['mse'] for m in volume_metrics]),
        'psnr': np.std([m['psnr'] for m in volume_metrics]),
        'ssim': np.std([m['ssim'] for m in volume_metrics])
    }
    
    if volume_metrics and 'snr' in volume_metrics[0]:
        std_metrics['snr'] = np.std([m.get('snr', 0) for m in volume_metrics])
        std_metrics['signal_power'] = np.std([m.get('signal_power', 0) for m in volume_metrics])
        std_metrics['noise_power'] = np.std([m.get('noise_power', 0) for m in volume_metrics])
    
    logger.info("\nGeneration Complete!")
    logger.info(f"Total samples generated: {len(select_idxs)}")
    logger.info(f"Average generation time per sample: {np.mean(generation_times):.2f}s")
    logger.info(f"Average condition generation time: {np.mean(condition_times):.2f}s")
    logger.info(f"Average volume metrics across all samples:")
    logger.info(f"  - MSE: {avg_metrics['mse']:.8f} ± {std_metrics['mse']:.8f}")
    logger.info(f"  - PSNR: {avg_metrics['psnr']:.8f} ± {std_metrics['psnr']:.8f} dB")
    logger.info(f"  - SSIM: {avg_metrics['ssim']:.8f} ± {std_metrics['ssim']:.8f}")
    if 'snr' in avg_metrics:
        logger.info(f"  - SNR: {avg_metrics['snr']:.8f} ± {std_metrics['snr']:.8f} dB")
        logger.info(f"  - Signal Power: {avg_metrics['signal_power']:.8f} ± {std_metrics['signal_power']:.8f}")
        logger.info(f"  - Noise Power: {avg_metrics['noise_power']:.8f} ± {std_metrics['noise_power']:.8f}")
    logger.info(f"Total time: {sum(generation_times)/60:.8f}min")
    logger.info(f"All results saved in: {output_base}")
    
    save_final_metrics(avg_metrics, std_metrics, generation_times, condition_times, len(select_idxs), output_base)


def main() -> None:
    """Main function for X-ray conditional sampling."""
    args = parse_args()
    config = load_config(os.path.join("configs", args.config))
    device = setup_device(config)
    
    output_base, debug_dir, samples_dir, conditions_dir, xray_verification_dir, logger = initialize_paths(args, config, device)
    if not args.verbose:
        logging.disable(logging.CRITICAL)
        logging.getLogger("LIDCVolumes").disabled = True

    with quiet_stdout(not args.verbose):
        model = load_model(config, args, device, logger)
    diffusion = setup_diffusion(args)
    with quiet_stdout(not args.verbose):
        dataset = setup_data(config, device, logger, debug_dir)
    
    generate_conditional_samples(
        dataset, model, diffusion, 
        args, config, device, logger, output_base,
        samples_dir, xray_verification_dir
    )


if __name__ == "__main__":
    main()

"""
Commands:
    # Single GPU sampling (default - saves all intermediate files):
    python sample_xrays.py --config lidc_stage2_global.yaml --ckpt results/001-DiT-XL-12-12/checkpoints/019000.pt --num-samples 1 --output-dir xrays_samples --new
    python sample_xrays.py --config lidc_stage2_global.yaml --ckpt results/001-DiT-XL-12-12/checkpoints/004000.pt --num-samples 1 --output-dir xrays_samples --new

    # Metrics-only mode (no intermediate files, for random seed experiments):
    python sample_xrays.py --config lidc_stage2_global.yaml --ckpt results/001-DiT-XL-12-12/checkpoints/019000.pt --num-samples 100 --output-dir metrics_only --new --no-save-intermediate
    
    # Metrics-only mode but save PNGs (no NIfTI files):
    python sample_xrays.py --config lidc_stage2_global.yaml --ckpt results/001-DiT-XL-12-12/checkpoints/019000.pt --num-samples 100 --output-dir metrics_with_png --new --no-save-intermediate --save-png
    
    # Metrics-only mode but save NIfTI files (no PNGs):
    python sample_xrays.py --config lidc_stage2_global.yaml --ckpt results/001-DiT-XL-12-12/checkpoints/019000.pt --num-samples 100 --output-dir metrics_with_nifti --new --no-save-intermediate --save-nifti

    # Distributed sampling:
    torchrun --nproc_per_node=8 sample_xrays.py --config configs/dit_base.py --num-samples 4 --output-dir samples --distributed

    # Notes:
    # - Use --no-save-intermediate to skip saving intermediate images/volumes (only compute metrics)
    # - When --no-save-intermediate is set, use --save-png and/or --save-nifti to selectively enable saving
    # - Final metrics are ALWAYS saved to final_metrics.txt and final_metrics.json

ckpt12:
/data/gpfs/projects/punim1874/zzk/stg3_projects/DeDiff_Transf/18-June/128/LIDC/12/B-768/12-b/results/010-DiT-B-12-12/checkpoints/best/0.0015.pt
"""
