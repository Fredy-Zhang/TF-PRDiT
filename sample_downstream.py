"""Downstream conditional sampling from a pretrained 3D DiT model.

Supported tasks:
- Volumetric infilling (infilling)
- Super-resolution (super_resolution)
- Deblurring (deblurring)
"""
import argparse
from contextlib import contextmanager, redirect_stdout
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import nibabel as nib
import numpy as np
import torch
import torch as th
from tqdm import tqdm

from conds.utils import compute_volume_metrics, create_val_loader, initialize_paths
from datasets.lidc import data_transform_backward
from utils.download import find_model
from models import DiT_models
from util import (
    evaluation_metrics,
    load_config,
    normalize_image,
    remove_empty_directories,
    save_evaluation_samples,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEBUG = True


@contextmanager
def quiet_stdout(enabled: bool):
    """Suppress noisy third-party/task diagnostics while keeping progress bars."""
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as sink, redirect_stdout(sink):
        yield

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Downstream conditional sampling")
    
    # Task configuration
    parser.add_argument("--task", type=str, required=True,
                       choices=["infilling", "super_resolution", "deblurring"],
                       help="Conditional task type")
    
    # Model and data
    parser.add_argument("--config", type=str, default="lidc_stage2_global.yaml", help="Config filename under configs/")
    parser.add_argument("--ckpt", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--num-samples", type=int, default=102, help="Number of samples")
    parser.add_argument("--num-sampling-steps", type=int, default=1000, help="Sampling steps")
    parser.add_argument("--output-dir", type=str, default="outputs_Cond", help="Output directory")
    
    # Task-specific arguments
    parser.add_argument("--mask-ratio", type=float, default=0.5, help="[Infilling] Ratio of masked regions")
    parser.add_argument("--mask-type", type=str, default="center", 
                       choices=["center", "random", "slices"],
                       help="[Infilling] Type of mask")
    parser.add_argument("--scale-factor", type=int, default=4, 
                       help="[Super-resolution] Downsampling scale factor")
    parser.add_argument("--blur-kernel-size", type=int, default=5,
                       help="[Deblurring] Gaussian blur kernel size (must be odd)")
    parser.add_argument("--blur-sigma", type=float, default=2.0,
                       help="[Deblurring] Gaussian blur sigma")
    # Output options
    parser.add_argument("--num-save-samples", type=int, default=102, 
                       help="Number of samples to save to disk")
    parser.add_argument("--no-save-intermediate", action="store_true", 
                       help="Disable saving intermediate images and volumes")
    parser.add_argument("--save-nifti", action="store_true", help="Save NIfTI files")
    parser.add_argument("--save-png", action="store_true", help="Save PNG images")
    parser.add_argument("--verbose", action="store_true",
                       help="Show model, dataset, and per-step diagnostic logs")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    return args


def setup_device(config: Dict[str, Any]) -> th.device:
    """Setup device and random seed."""
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    th.manual_seed(config.training.seed)
    return device


def load_model(config: Dict[str, Any], args: argparse.Namespace, 
               device: th.device, logger: logging.Logger) -> th.nn.Module:
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


def setup_diffusion(args: argparse.Namespace, logger: logging.Logger):
    """Setup diffusion process based on task type."""
    if args.task == "infilling":
        from diffusion.inpainting import IaNDiffusion
        logger.info("Using volumetric infilling diffusion")
    elif args.task == "super_resolution":
        from diffusion.super_resolution import IaNDiffusion
        logger.info("Using super-resolution diffusion")
    elif args.task == "deblurring":
        from diffusion.deblurring import IaNDiffusion
        logger.info("Using deblurring diffusion")
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    diffusion = IaNDiffusion(timestep_respacing=str(args.num_sampling_steps), loss_type="l2")
    diffusion.config.debug = args.verbose
    diffusion.config.save_detailed = args.verbose
    return diffusion


def setup_data(config: Dict[str, Any], device: th.device, 
               logger: logging.Logger, debug_dir: str) -> Any:
    """Load validation dataset."""
    dataset = create_val_loader(config)
    logger.info(f"Loaded dataset with {len(dataset)} samples")
    return dataset


def create_inpainting_mask(
    shape: Tuple[int, ...], 
    mask_type: str = "center",
    mask_ratio: float = 0.5,
    device: th.device = None
) -> th.Tensor:
    """
    Create inpainting mask for a single volume.
    
    Args:
        shape: Shape [C, D, H, W]
        mask_type: 'center', 'random', or 'slices'
        mask_ratio: Ratio of masked (unknown) regions
        device: Device for tensor
    
    Returns:
        mask: Binary mask [1, D, H, W] (1=valid, 0=missing)
    """
    C, D, H, W = shape
    mask = th.ones(1, D, H, W, device=device)
    
    if mask_type == "center":
        # Mask out center region
        d_size = int(D * mask_ratio)
        h_size = int(H * mask_ratio)
        w_size = int(W * mask_ratio)
        
        d_start = (D - d_size) // 2
        h_start = (H - h_size) // 2
        w_start = (W - w_size) // 2
        
        mask[:, d_start:d_start+d_size, h_start:h_start+h_size, w_start:w_start+w_size] = 0
        
    elif mask_type == "random":
        # Random voxel masking
        num_voxels = D * H * W
        num_masked = int(num_voxels * mask_ratio)
        indices = th.randperm(num_voxels, device=device)[:num_masked]
        
        flat_mask = mask.view(1, -1)
        flat_mask[:, indices] = 0
        mask = flat_mask.view(1, D, H, W)
        
    elif mask_type == "slices":
        # Remove middle slices
        num_masked_slices = int(D * mask_ratio)
        start_slice = (D - num_masked_slices) // 2
        mask[:, start_slice:start_slice+num_masked_slices, :, :] = 0
    
    return mask


def prepare_conditions(
    datarow: th.Tensor,
    args: argparse.Namespace,
    idx: int,
    device: th.device
) -> Tuple[Optional[th.Tensor], Optional[th.Tensor]]:
    if args.task == "infilling":
        # Move datarow to device first
        datarow = datarow.to(device)
        
        # Create masked image and mask
        mask = create_inpainting_mask(
            datarow.shape,
            mask_type=args.mask_type,
            mask_ratio=args.mask_ratio,
            device=device
        )
        # Masked image (corrupted input)
        masked_image = datarow * mask
        return masked_image, mask
        
    elif args.task == "super_resolution":
        # Move datarow to device first
        datarow = datarow.to(device)
        
        # Downsample as condition
        C, D, H, W = datarow.shape
        scale = args.scale_factor
        low_res = th.nn.functional.interpolate(
            datarow.unsqueeze(0),  # Add batch dimension [1, C, D, H, W]
            size=(D//scale, H//scale, W//scale),
            mode='trilinear',
            align_corners=False
        ).squeeze(0)  # Remove batch dimension [C, D, H, W]
        return low_res, None
    
    elif args.task == "deblurring":
        # Move datarow to device first
        datarow = datarow.to(device)
        
        # Apply Gaussian blur as condition
        from diffusion.deblurring import apply_blur_3d
        blurred = apply_blur_3d(
            datarow.unsqueeze(0),  # [1, C, D, H, W]
            kernel_size=args.blur_kernel_size,
            sigma=args.blur_sigma
        ).squeeze(0)  # [C, D, H, W]
        return blurred, None
    
    else:
        raise ValueError(f"Unknown task: {args.task}")


def generate_diffusion_samples(
    model: th.nn.Module,
    diffusion,
    conditions: th.Tensor,
    mask: Optional[th.Tensor],
    idx: int,
    datarow: th.Tensor,
    args: argparse.Namespace,
    config: Dict[str, Any],
    device: th.device,
    sample_dir: Optional[str]
) -> th.Tensor:
    """Run diffusion sampling for different tasks."""
    z = th.randn(
        1, 
        config.model.in_channels, 
        config.data.image_size, 
        config.data.image_size, 
        config.data.image_size, 
        device=device
    )

    # Prepare kwargs based on task
    sample_kwargs = {
        "model": model.forward,
        "z": z,
        "idx": idx,
        "ref_vol": datarow[0].to(device),
        "device": device,
        "new_sampling": True,
        "sample_dir": sample_dir,
        "model_kwargs": {"y": None} if config.model.num_classes else {}
    }
    
    if args.task == "infilling":
        sample_kwargs["conditions"] = conditions  # masked image
        sample_kwargs["mask"] = mask
    elif args.task == "super_resolution":
        sample_kwargs["conditions"] = conditions  # low-res image
    elif args.task == "deblurring":
        sample_kwargs["conditions"] = conditions  # blurred image
    
    samples = diffusion.p_sample_loop(**sample_kwargs)
    
    return samples


def generate_conditional_samples(
    dataset: Any,
    model: th.nn.Module,
    diffusion,
    args: argparse.Namespace,
    config: Dict[str, Any],
    device: th.device,
    logger: logging.Logger,
    output_base: str,
    samples_dir: str,
    verification_dir: str
) -> None:
    """Generate conditional samples for different tasks."""
    generation_times = []
    condition_times = []
    volume_metrics = []
    
    select_idxs = range(len(dataset))[:args.num_samples]
    num_save_samples = min(args.num_save_samples, len(select_idxs))
    
    # Determine save flags
    if args.no_save_intermediate:
        save_intermediate = False
        save_nifti = args.save_nifti
        save_png = args.save_png
        logger.info(f"Processing {len(select_idxs)} samples, computing metrics ONLY")
    else:
        save_intermediate = True
        save_nifti = True
        save_png = True
        logger.info(f"Processing {len(select_idxs)} samples, saving intermediate results")
    
    for idx in tqdm(select_idxs, desc=f"Generating {args.task} samples"):
        start_time = time.time()
        datarow = dataset[idx]["image"]
        sample_id = dataset[idx].get("id", f"sample_{idx}")
        sanitized_id = sanitize_id(sample_id) if sample_id else f"sample_{idx}"
        
        if idx == 0:
            logger.info(f"Sample ID format: '{sample_id}' -> '{sanitized_id}'")
        
        # Prepare task-specific conditions
        cond_start = time.time()
        conditions, mask = prepare_conditions(datarow, args, idx, device)
        condition_times.append(time.time() - cond_start)
        
        should_save_intermediate = save_intermediate and (idx < num_save_samples)
        sample_dir = create_sample_directories(
            samples_dir, sample_id, idx
        ) if should_save_intermediate else None
        
        # Save conditions for verification
        if should_save_intermediate and DEBUG:
            save_dir = os.path.join(verification_dir, sanitized_id)
            os.makedirs(save_dir, exist_ok=True)
            save_task_conditions(
                conditions, mask, datarow, save_dir, args.task, save_png=save_png
            )
        
        # Generate samples
        with quiet_stdout(not args.verbose):
            samples = generate_diffusion_samples(
                model, diffusion, conditions, mask,
                idx, datarow, args, config, device, sample_dir
            )
        
        # Save results
        if should_save_intermediate:
            save_reference_data(datarow, sample_dir, save_nifti=save_nifti, save_png=save_png)
            save_generated_samples(samples, sample_dir, config, logger, 
                                 save_nifti=save_nifti, save_png=save_png)
            
            # Save individual comparison images for paper figures
            indiv_dir = os.path.join(sample_dir, "individual_slices")
            save_individual_comparison(
                estimated=samples,
                conditions=conditions,
                reference=datarow,
                save_dir=indiv_dir,
                task=args.task,
                args=args,
            )
        
        # Compute metrics
        samples = data_transform_backward(samples)
        samples = th.clamp(samples, 0.0, 1.0)
        datarow = data_transform_backward(datarow)
        
        metrics = evaluation_metrics(datarow, samples, verbose=args.verbose)
        volume_metrics.append(metrics)
        
        generation_time = time.time() - start_time
        generation_times.append(generation_time)
        
        if sample_dir:
            save_sample_metrics(metrics, sample_dir, idx, generation_time, 
                              condition_times[-1] if condition_times else 0)
            remove_empty_directories(sample_dir)
        
        log_sample_progress(idx, select_idxs, generation_times, 
                          condition_times, metrics, logger)
    
    log_final_statistics(select_idxs, generation_times, condition_times, 
                        volume_metrics, output_base, logger)
    remove_empty_directories(output_base)


def save_task_conditions(
    conditions: th.Tensor,
    mask: Optional[th.Tensor],
    datarow: th.Tensor,
    save_dir: str,
    task: str,
    save_png: bool = True
) -> None:
    """Save task-specific conditions for verification."""
    if not save_png:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    if task == "infilling":
        # Save masked image and mask
        masked_np = conditions.cpu().squeeze().numpy()
        mask_np = mask.cpu().squeeze().numpy()
        
        # Save middle slices
        D, H, W = masked_np.shape
        save_slice_png(masked_np[D//2], os.path.join(save_dir, "masked_axial.png"))
        save_slice_png(mask_np[D//2], os.path.join(save_dir, "mask_axial.png"))
        
        # Save 3D arrays
        np.save(os.path.join(save_dir, "masked_image.npy"), masked_np)
        np.save(os.path.join(save_dir, "mask.npy"), mask_np)
    
    elif task == "super_resolution":
        # Save low-res condition
        low_res_np = conditions.cpu().squeeze().numpy()
        D, H, W = low_res_np.shape
        save_slice_png(low_res_np[D//2], os.path.join(save_dir, "low_res_axial.png"))
        np.save(os.path.join(save_dir, "low_res.npy"), low_res_np)
    
    elif task == "deblurring":
        # Save blurred condition and original
        blurred_np = conditions.cpu().squeeze().numpy()
        D, H, W = blurred_np.shape
        save_slice_png(blurred_np[D//2], os.path.join(save_dir, "blurred_axial.png"))
        np.save(os.path.join(save_dir, "blurred_image.npy"), blurred_np)
        
        # Save original for comparison
        original_np = datarow.cpu().squeeze().numpy()
        save_slice_png(original_np[D//2], os.path.join(save_dir, "original_axial.png"))
        np.save(os.path.join(save_dir, "original_image.npy"), original_np)


def save_slice_png(slice_data: np.ndarray, filename: str) -> None:
    """Save a 2D slice as PNG."""
    slice_norm = normalize_image(slice_data)
    slice_uint8 = (slice_norm * 255).astype(np.uint8)
    cv2.imwrite(filename, slice_uint8)


def save_individual_comparison(
    estimated: th.Tensor,
    conditions: th.Tensor,
    reference: th.Tensor,
    save_dir: str,
    task: str,
    args: argparse.Namespace,
) -> None:
    """
    Save each comparison panel as an individual image file.
    
    Output structure in save_dir:
        estimated_sagittal.png, estimated_coronal.png, estimated_axial.png
        condition_sagittal.png, condition_coronal.png, condition_axial.png
        reference_sagittal.png, reference_coronal.png, reference_axial.png
        difference_sagittal.png, difference_coronal.png, difference_axial.png
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    def _to_3d_np(t):
        t = t.detach().cpu().float()
        if t.ndim == 5: t = t[0, 0]
        elif t.ndim == 4: t = t[0]
        return t.numpy()

    def _norm(s):
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo + 1e-8)

    est_np = _to_3d_np(estimated)
    ref_np = _to_3d_np(reference)
    D, H, W = est_np.shape

    # Prepare condition volume (handle different sizes for SR)
    cond = conditions.detach().cpu().float()
    if cond.ndim == 5: cond = cond[0, 0]
    elif cond.ndim == 4: cond = cond[0]
    # For super-resolution: upsample condition to match estimated size
    if cond.shape != est_np.shape:
        import torch.nn.functional as F_interp
        cond_up = F_interp.interpolate(
            cond.unsqueeze(0).unsqueeze(0),
            size=(D, H, W), mode='trilinear', align_corners=False
        )[0, 0]
        cond_np = cond_up.numpy()
    else:
        cond_np = cond.numpy()

    views = {
        'sagittal': (lambda v: v[:, :, W // 2]),
        'coronal':  (lambda v: v[:, H // 2, :]),
        'axial':    (lambda v: v[D // 2, :, :]),
    }

    # Determine condition label based on task
    cond_labels = {
        'infilling': 'masked',
        'super_resolution': 'lowres_upsampled',
        'deblurring': 'blurred',
    }
    cond_label = cond_labels.get(task, 'condition')

    for view_name, slice_fn in views.items():
        est_s = _norm(slice_fn(est_np))
        cond_s = _norm(slice_fn(cond_np))
        ref_s = _norm(slice_fn(ref_np))
        diff_s = np.abs(est_s - ref_s)

        # Save estimated
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(est_s, cmap='gray'); ax.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(save_dir, f'estimated_{view_name}.png'),
                    bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()

        # Save condition
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(cond_s, cmap='gray'); ax.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(save_dir, f'{cond_label}_{view_name}.png'),
                    bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()

        # Save reference
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(ref_s, cmap='gray'); ax.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(save_dir, f'reference_{view_name}.png'),
                    bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()

        # Save difference (hot colormap, with colorbar)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        im = ax.imshow(diff_s, cmap='hot'); ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(save_dir, f'difference_{view_name}.png'),
                    bbox_inches='tight', pad_inches=0.02, dpi=150)
        plt.close()


def sanitize_id(sample_id: str) -> str:
    """Sanitize sample ID for filesystem use."""
    path_parts = sample_id.split(os.sep)
    for part in reversed(path_parts):
        if part.startswith('LIDC-IDRI-'):
            return part
    
    basename = os.path.basename(sample_id)
    for ext in ['.h5', '.nii.gz', '.nii', '.hdf5']:
        if basename.endswith(ext):
            basename = basename[:-len(ext)]
            break
    
    common_suffixes = [
        '_ct_xray_data', '_ct_128_norm', '_ct_256_norm',
        '_ct_64_norm', '_ct_norm', '_norm', '_ct',
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


def create_sample_directories(output_base: str, sample_id: str, idx: int = None) -> str:
    """Create sample directories."""
    sanitized_id = sanitize_id(sample_id) if sample_id else f"sample_{idx}"
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


def save_reference_data(datarow: th.Tensor, sample_dir: str, 
                       save_nifti: bool = True, save_png: bool = True) -> None:
    """Save reference CT volume."""
    if not save_nifti and not save_png:
        return
    
    reference_dir = os.path.join(sample_dir, "reference")
    os.makedirs(reference_dir, exist_ok=True)
    
    if save_nifti:
        img = nib.Nifti1Image(datarow[0].numpy(), np.eye(4))
        nib.save(img, os.path.join(reference_dir, "original_ct.nii.gz"))
    
    if save_png:
        save_middle_slices_as_png(datarow, reference_dir)


def save_middle_slices_as_png(volume: th.Tensor, save_dir: str) -> None:
    """Save middle slices as PNG."""
    os.makedirs(save_dir, exist_ok=True)
    
    vol_np = volume.cpu().numpy() if isinstance(volume, th.Tensor) else np.asarray(volume)
    if vol_np.ndim == 5:
        vol_np = vol_np[0]
    if vol_np.ndim == 4:
        vol_np = vol_np[0]
    
    D, H, W = vol_np.shape
    slices = {
        'axial': vol_np[D//2, :, :],
        'coronal': vol_np[:, H//2, :],
        'sagittal': vol_np[:, :, W//2]
    }
    
    for name, slice_data in slices.items():
        save_slice_png(slice_data, os.path.join(save_dir, f"middle_slice_{name}.png"))


def save_generated_samples(samples: th.Tensor, sample_dir: str, 
                          config: Dict[str, Any], logger: logging.Logger,
                          save_nifti: bool = True, save_png: bool = True) -> None:
    """Save generated samples."""
    if not save_nifti and not save_png:
        return
    
    generated_dir = os.path.join(sample_dir, "generated")
    os.makedirs(generated_dir, exist_ok=True)
    
    samples_normalized = data_transform_backward(samples)
    samples_normalized = th.clamp(samples_normalized, 0.0, 1.0)
    
    if save_nifti:
        save_evaluation_samples(
            samples_normalized, generated_dir, 
            config.data.image_size, epoch=0, logger=logger
        )
    
    if save_png:
        save_middle_slices_as_png(samples_normalized, generated_dir)


def save_sample_metrics(metrics: Dict[str, float], sample_dir: str, 
                       idx: int, generation_time: float, condition_time: float) -> None:
    """Save evaluation metrics for a sample."""
    metrics_file = os.path.join(sample_dir, "metrics.txt")
    metrics_json = os.path.join(sample_dir, "metrics.json")
    
    with open(metrics_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Evaluation Metrics for Sample {idx}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generation Time: {generation_time:.4f} s\n")
        f.write(f"Condition Time: {condition_time:.4f} s\n\n")
        f.write(f"MSE: {metrics['mse']:.8f}\n")
        f.write(f"PSNR: {metrics['psnr']:.8f} dB\n")
        f.write(f"SSIM: {metrics['ssim']:.8f}\n")
    
    metrics_dict = {
        "sample_index": int(idx),
        "generation_time": float(generation_time),
        "condition_time": float(condition_time),
        **{k: float(v) for k, v in metrics.items()}
    }
    
    with open(metrics_json, "w") as f:
        json.dump(metrics_dict, f, indent=2)


def log_sample_progress(idx: int, select_idxs: range, generation_times: List[float],
                       condition_times: List[float], metrics: Dict[str, float], 
                       logger: logging.Logger) -> None:
    """Log progress for current sample."""
    avg_time = np.mean(generation_times)
    eta = avg_time * (len(select_idxs) - idx - 1)
    
    logger.info(f"Completed {idx+1}/{len(select_idxs)}")
    logger.info(f"  Gen: {generation_times[-1]:.2f}s, Cond: {condition_times[-1]:.2f}s")
    logger.info(f"  MSE: {metrics['mse']:.6f}, PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.4f}")
    logger.info(f"  ETA: {eta/60:.2f}min")


def log_final_statistics(select_idxs: range, generation_times: List[float],
                        condition_times: List[float], volume_metrics: List[Dict[str, float]],
                        output_base: str, logger: logging.Logger) -> None:
    """Log final statistics."""
    avg_metrics = {
        'mse': np.mean([m['mse'] for m in volume_metrics]),
        'psnr': np.mean([m['psnr'] for m in volume_metrics]),
        'ssim': np.mean([m['ssim'] for m in volume_metrics])
    }
    std_metrics = {
        'mse': np.std([m['mse'] for m in volume_metrics]),
        'psnr': np.std([m['psnr'] for m in volume_metrics]),
        'ssim': np.std([m['ssim'] for m in volume_metrics])
    }
    
    logger.info("\nGeneration Complete!")
    logger.info(f"Total samples: {len(select_idxs)}")
    logger.info(f"Avg time: {np.mean(generation_times):.2f}s")
    logger.info(f"MSE: {avg_metrics['mse']:.8f} ± {std_metrics['mse']:.8f}")
    logger.info(f"PSNR: {avg_metrics['psnr']:.8f} ± {std_metrics['psnr']:.8f} dB")
    logger.info(f"SSIM: {avg_metrics['ssim']:.8f} ± {std_metrics['ssim']:.8f}")
    
    # Save final metrics
    final_file = os.path.join(output_base, "final_metrics.json")
    with open(final_file, "w") as f:
        json.dump({
            "num_samples": len(select_idxs),
            "avg_generation_time": float(np.mean(generation_times)),
            "avg_condition_time": float(np.mean(condition_times)),
            "metrics": {
                "mean": {k: float(v) for k, v in avg_metrics.items()},
                "std": {k: float(v) for k, v in std_metrics.items()}
            }
        }, f, indent=2)


def main() -> None:
    """Main function for conditional sampling."""
    args = parse_args()
    config = load_config(os.path.join("configs", args.config))
    device = setup_device(config)
    
    output_base, debug_dir, samples_dir, conditions_dir, verification_dir, logger = \
        initialize_paths(args, config, device)

    if not args.verbose:
        logging.disable(logging.CRITICAL)
        logging.getLogger("LIDCVolumes").disabled = True
    
    logger.info(f"Task: {args.task}")
    with quiet_stdout(not args.verbose):
        model = load_model(config, args, device, logger)
    diffusion = setup_diffusion(args, logger)
    with quiet_stdout(not args.verbose):
        dataset = setup_data(config, device, logger, debug_dir)
    
    generate_conditional_samples(
        dataset, model, diffusion, 
        args, config, device, logger, output_base,
        samples_dir, verification_dir
    )


if __name__ == "__main__":
    main()
