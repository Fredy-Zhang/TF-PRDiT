import os
import logging
import random
import yaml
from collections import OrderedDict
from datetime import datetime
from glob import glob
from typing import Any, Tuple, Optional, Dict

import numpy as np
import nibabel as nib
import torch
import torch.distributed as dist
import wandb
from skimage.metrics import structural_similarity as SSIM
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets import get_voxel_dataset
from datasets.lidc import (GLOBAL_MEAN, 
                           GLOBAL_STD, 
                           data_transform_forward, 
                           data_transform_backward)


def remove_empty_directories(root: str) -> None:
    """Remove empty output directories recursively, keeping every saved file."""
    if not os.path.isdir(root):
        return
    for current_dir, _, _ in os.walk(root, topdown=False):
        if current_dir == root:
            continue
        try:
            os.rmdir(current_dir)
        except OSError:
            # The directory contains an output or was already removed.
            pass


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range."""
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        return (image - img_min) / (img_max - img_min)
    return image


def tensor_back_to_unMinMax(input_image, min, max):
    """Convert tensor from normalized range back to original range."""
    return input_image * (max - min) + min


def Peak_Signal_to_Noise_Rate_3D(arr1, arr2, size_average=True, PIXEL_MAX=1.0):
    """Compute 3D PSNR metric."""
    assert isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray)
    assert arr1.ndim == 4 and arr2.ndim == 4
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)
    eps = 1e-10
    se = np.power(arr1 - arr2, 2)
    mse = se.mean(axis=1).mean(axis=1).mean(axis=1)
    zero_mse = np.where(mse == 0)
    mse[zero_mse] = eps
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    psnr[zero_mse] = 100
    return psnr.mean() if size_average else psnr


def Signal_to_Noise_Ratio_3D(arr1, arr2, size_average=True):
    """Compute 3D SNR metric."""
    assert isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray)
    assert arr1.ndim == 4 and arr2.ndim == 4
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)
    eps = 1e-10
    signal_power = np.power(arr1, 2).mean(axis=1).mean(axis=1).mean(axis=1)
    noise_power = np.power(arr1 - arr2, 2).mean(axis=1).mean(axis=1).mean(axis=1)
    zero_noise = np.where(noise_power == 0)
    noise_power[zero_noise] = eps
    snr = 10 * np.log10(signal_power / noise_power)
    snr[zero_noise] = 100.0
    if size_average:
        return snr.mean(), signal_power.mean(), noise_power.mean()
    return snr, signal_power, noise_power


def Structural_Similarity(arr1, arr2, size_average=True, PIXEL_MAX=1.0):
    """Compute 3D SSIM across depth, height, and width views."""
    assert isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray)
    assert arr1.ndim == 4 and arr2.ndim == 4
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)

    n = arr1.shape[0]
    arr1_d = np.transpose(arr1, (0, 2, 3, 1))
    arr2_d = np.transpose(arr2, (0, 2, 3, 1))
    ssim_d = np.asarray(
        [SSIM(arr1_d[i], arr2_d[i], data_range=PIXEL_MAX, channel_axis=2) for i in range(n)],
        dtype=np.float64,
    )

    arr1_h = np.transpose(arr1, (0, 1, 3, 2))
    arr2_h = np.transpose(arr2, (0, 1, 3, 2))
    ssim_h = np.asarray(
        [SSIM(arr1_h[i], arr2_h[i], data_range=PIXEL_MAX, channel_axis=2) for i in range(n)],
        dtype=np.float64,
    )

    ssim_w = np.asarray(
        [SSIM(arr1[i], arr2[i], data_range=PIXEL_MAX, channel_axis=0) for i in range(n)],
        dtype=np.float64,
    )
    ssim_avg = (ssim_d + ssim_h + ssim_w) / 3
    if size_average:
        return [ssim_d.mean(), ssim_h.mean(), ssim_w.mean(), ssim_avg.mean()]
    return [ssim_d, ssim_h, ssim_w, ssim_avg]


def evaluation_metrics(real_CT, generated_CT, verbose=False):
    """Compute MSE, PSNR, SSIM, and SNR for 3D CT volumes."""
    if isinstance(real_CT, torch.Tensor):
        real_CT = real_CT.cpu().numpy()
    if isinstance(generated_CT, torch.Tensor):
        generated_CT = generated_CT.cpu().numpy()

    while real_CT.ndim > 4:
        real_CT = np.squeeze(real_CT, axis=0)
    while generated_CT.ndim > 4:
        generated_CT = np.squeeze(generated_CT, axis=0)

    if verbose:
        print(f"Real CT shape: {real_CT.shape}, range: {real_CT.min()}, {real_CT.max()}")
        print(f"Generated CT shape: {generated_CT.shape}, range: {generated_CT.min()}, {generated_CT.max()}")

    ssim = Structural_Similarity(real_CT, generated_CT, size_average=False, PIXEL_MAX=1.0)[-1]
    if isinstance(ssim, np.ndarray):
        ssim = ssim.item() if ssim.size == 1 else ssim[0]
    mse = np.mean((real_CT - generated_CT) ** 2)

    generated_cts = tensor_back_to_unMinMax(generated_CT, 0, 2500).astype(np.int32)
    real_cts = tensor_back_to_unMinMax(real_CT, 0, 2500).astype(np.int32)

    psnr_3d = Peak_Signal_to_Noise_Rate_3D(real_cts, generated_cts, size_average=False, PIXEL_MAX=4095)
    if isinstance(psnr_3d, np.ndarray):
        psnr_3d = psnr_3d[0] if psnr_3d.size > 0 else psnr_3d.item()

    snr_3d, signal_power, noise_power = Signal_to_Noise_Ratio_3D(real_cts, generated_cts, size_average=False)
    if isinstance(snr_3d, np.ndarray):
        snr_3d = snr_3d[0] if snr_3d.size > 0 else snr_3d.item()
    if isinstance(signal_power, np.ndarray):
        signal_power = signal_power[0] if signal_power.size > 0 else signal_power.item()
    if isinstance(noise_power, np.ndarray):
        noise_power = noise_power[0] if noise_power.size > 0 else noise_power.item()

    return {
        "mse": mse,
        "psnr": psnr_3d,
        "ssim": ssim,
        "snr": snr_3d,
        "signal_power": signal_power,
        "noise_power": noise_power,
    }

def setup_torch_config():
    # Enable TF32 for faster training on A100 GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set default dtype to float32
    torch.set_default_dtype(torch.float32)
    
    # Configure CUDA settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    torch.backends.cuda.enable_flash_sdp(True)        # try to use FlashAttention  
    torch.backends.cuda.enable_math_sdp(False)    # disable the pure-C++ math path  
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    
def setup_wandb(config, rank) -> None:
    if rank == 0 and config.wandb.enable:
        print(f"\n{'='*80}", flush=True)
        print(f"🚀 Initializing Weights & Biases (wandb)", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"  Project: {config.wandb.project}", flush=True)
        print(f"  Entity:  {config.wandb.entity}", flush=True)
        print(f"  Model:   {config.model.name}", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        run_name = f"DiT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=f"{config.model.name}-{config.data.image_size}-base",
            config={
                "architecture": config.model.name,
                "image_size": config.data.image_size,
                "batch_size": config.training.batch_size,
                "learning_rate": config.training.learning_rate,
                "epochs": config.training.epochs,
                "num_workers": config.data.num_workers,
                "seed": config.training.seed,
            },
            tags=[config.wandb.tags]
        )   
    elif rank == 0 and not config.wandb.enable:
        print(f"\n⚠️  WandB logging is DISABLED (set wandb.enable: True in config to enable)\n", flush=True)

def to_uint8_img(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32))
    # per-slice min-max; replace with fixed windowing if you prefer (e.g., HU)
    x_min, x_max = x.min(), x.max()
    if x_max > x_min:
        x = (x - x_min) / (x_max - x_min)
    else:
        x = np.zeros_like(x)
    return (x * 255).astype(np.uint8)

def sample_from_model(
    ema_model: torch.nn.Module, 
    diffusion: Any,
    device: torch.device, 
    rank: int, 
    experiment_dir: str,
    image_size: int, 
    epoch: int, 
    logger: logging.Logger,
    num_samples: int = 4,
    batch_size: int = 2,
    train_steps: int = None,
    config: Any = None
) -> Optional[Dict[str, float]]:
    ema_model.eval()
    
    if isinstance(device, int):
        device = torch.device(f'cuda:{device}')
    elif isinstance(device, str):
        device = torch.device(device)
    
    if rank == 0:
        logger.info(f"Generating {num_samples} samples using EMA model with 1000 timesteps...")
    
    # Create sampling diffusion with 1000 timesteps for higher quality
    from diffusion.image_noise_diffusion import IaNDiffusion
    sampling_diffusion = IaNDiffusion(timestep_respacing=1000, loss_type="l2")
    
    # Create sampling directory
    sample_dir = os.path.join(experiment_dir, f"samples_epoch_{epoch}")
    os.makedirs(sample_dir, exist_ok=True)
    
    with torch.no_grad():
        # Generate samples in batches
        for batch_idx in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - batch_idx)
            
            # Create noise tensor for sampling
            z = torch.randn(
                current_batch_size,
                1,  # in_channels
                image_size,
                image_size,
                image_size,
                device=device
            )
            
            # Generate samples using diffusion sampling with 1000 timesteps
            xs_samples, x0_samples = sampling_diffusion.p_sample_loop(
                ema_model.forward,
                z.shape,
                z,
                new_sampling=True,  # Use new sampling method
                model_kwargs={}
            )
            
            # Apply backward transformation to convert from [-1, 1] to [0, 1]
            xs_samples = data_transform_backward(xs_samples[-1])
            x0_samples = data_transform_backward(x0_samples)
            xs_samples = torch.clamp(xs_samples, 0.0, 1.0)
            x0_samples = torch.clamp(x0_samples, 0.0, 1.0)
            
            if rank == 0:
                logger.info(f"Generated batch {batch_idx//batch_size + 1}: "
                           f"XS range: [{xs_samples.min().item():.4f}, {xs_samples.max().item():.4f}], "
                           f"X0 range: [{x0_samples.min().item():.4f}, {x0_samples.max().item():.4f}]")
                
                # Save generated samples
                save_evaluation_samples(
                    samples=xs_samples,
                    experiment_dir=os.path.join(sample_dir, "xs"),
                    image_size=image_size,
                    epoch=batch_idx,
                    nii_number=current_batch_size,
                    logger=logger
                )
                
                save_evaluation_samples(
                    samples=x0_samples,
                    experiment_dir=os.path.join(sample_dir, "x0"),
                    image_size=image_size,
                    epoch=batch_idx,
                    nii_number=current_batch_size,
                    logger=logger
                )
                
                if wandb.run is not None:
                    try:
                        x0_axial = to_uint8_img(x0_samples[0,0,64,:,:].cpu().numpy())
                        x0_coronal = to_uint8_img(x0_samples[0,0,:,64,:].cpu().numpy())
                        x0_sagittal = to_uint8_img(x0_samples[0,0,:,:,64].cpu().numpy())
                        xs_axial = to_uint8_img(xs_samples[0,0,64,:,:].cpu().numpy())
                        xs_coronal = to_uint8_img(xs_samples[0,0,:,64,:].cpu().numpy())
                        xs_sagittal = to_uint8_img(xs_samples[0,0,:,:,64].cpu().numpy())
                        # Use train_steps if available, otherwise fall back to epoch
                        log_step = train_steps if train_steps is not None else epoch
                        wandb.log({
                            "val/x0_axial": wandb.Image(x0_axial, caption=f"epoch {epoch}"),
                            "val/x0_coronal": wandb.Image(x0_coronal, caption=f"epoch {epoch}"),
                            "val/x0_sagittal": wandb.Image(x0_sagittal, caption=f"epoch {epoch}"),
                            "val/xs_axial": wandb.Image(xs_axial, caption=f"epoch {epoch}"),
                            "val/xs_coronal": wandb.Image(xs_coronal, caption=f"epoch {epoch}"),
                            "val/xs_sagittal": wandb.Image(xs_sagittal, caption=f"epoch {epoch}"),
                            "epoch": epoch
                        }, step=log_step)
                    except Exception as e:
                        if rank == 0:
                            logger.warning(f"Failed to log sampling images to wandb: {e}")
    
    if rank == 0:
        logger.info(f"✅ Generated {num_samples} samples and saved to {sample_dir}")
    
    ema_model.eval()  # Keep EMA in eval mode

def evaluate_model_depth_zero(
    model: torch.nn.Module, val_loader, diffusion: Any,
    device: torch.device, rank: int, experiment_dir: str,
    image_size: int, epoch: int, logger: logging.Logger,
    train_steps: int = None
) -> Dict[str, float]:
    
    model.eval()

    # Convert device to torch.device if it's an integer
    if isinstance(device, int):
        device = torch.device(f'cuda:{device}')
    elif isinstance(device, str):
        device = torch.device(device)
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            x = batch["image"].to(device)
            t = torch.full((x.shape[0],), 100, device=device)
            samples = diffusion.q_sample(x, t)
            
            # For depth=0, use full precision to avoid any potential issues
            model_output = model(samples, t)
        
            # If model returns tuple (happens with return_intermediate=True), get just the output
            if isinstance(model_output, tuple):
                model_output = model_output[0]
            
            # Calculate metrics based on model output
            if model_output.shape[1] > x.shape[1]:  # Model outputs both noise and image predictions
                eps_recon, img_recon = model_output.chunk(2, dim=1)
            else:  # Model outputs only image predictions
                img_recon = model_output
            img_recon = data_transform_backward(img_recon)
            # Save main results from the first batch
            if i == 0:
                save_evaluation_samples(img_recon, 
                                        experiment_dir, 
                                        image_size, epoch, 2, logger)
                save_evaluation_samples(x, 
                                        f"{experiment_dir}/gt", 
                                        image_size, epoch, 2, logger)
                if wandb.run is not None:
                    try:
                        x0_axial = to_uint8_img(img_recon[0,0,64,:,:].cpu().numpy())
                        x0_coronal = to_uint8_img(img_recon[0,0,:,64,:].cpu().numpy())
                        x0_sagittal = to_uint8_img(img_recon[0,0,:,:,64].cpu().numpy())
                        # Use train_steps if available, otherwise fall back to epoch
                        log_step = train_steps if train_steps is not None else epoch
                        wandb.log({
                            "val/axial": wandb.Image(x0_axial, caption=f"epoch {epoch}"),
                            "val/coronal": wandb.Image(x0_coronal, caption=f"epoch {epoch}"),
                            "val/sagittal": wandb.Image(x0_sagittal, caption=f"epoch {epoch}"),
                            "epoch": epoch
                        }, step=log_step)
                    except Exception as e:
                        if logger:
                            logger.warning(f"Failed to log validation images to wandb: {e}")
                
                break

    model.train()


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Look for checkpoint files with 6-digit epoch naming (XXXXXX.pt)
    checkpoint_pattern = os.path.join(checkpoint_dir, "*.pt")
    checkpoint_files = glob(checkpoint_pattern)
    
    if not checkpoint_files:
        return None
    
    # Extract epoch numbers from filenames and find the latest
    latest_checkpoint = None
    latest_epoch = -1
    
    for checkpoint_file in checkpoint_files:
        filename = os.path.basename(checkpoint_file)
        # Extract epoch number from filename (XXXXXX.pt format)
        if filename.endswith('.pt') and len(filename) == 9:  # 6 digits + .pt
            try:
                epoch_str = filename[:-3]  # Remove .pt extension
                epoch_num = int(epoch_str)
                if epoch_num > latest_epoch:
                    latest_epoch = epoch_num
                    latest_checkpoint = checkpoint_file
            except ValueError:
                continue  # Skip files that don't match the naming pattern
    
    return latest_checkpoint

def load_checkpoint_state(checkpoint_path: str, model, ema, optimizer, device, rank: int = 0, logger=None):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
        
        # Load EMA state
        ema.load_state_dict(checkpoint['ema'])
        
        # Load optimizer state if available
        optimizer_loaded = False
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer_loaded = True
        
        # Get training state
        start_epoch = checkpoint.get('epoch', 0)
        resume_train_steps = checkpoint.get('train_steps', 0)
        last_eval_loss = checkpoint.get('loss', None)
        
        if rank == 0 and logger:
            logger.info(f"✅ Successfully loaded checkpoint from epoch {start_epoch}")
            logger.info(f"   Resuming from training step {resume_train_steps}")
            if last_eval_loss is not None:
                logger.info(f"   Last evaluation loss: {last_eval_loss:.6f}")
            
            if optimizer_loaded:
                logger.info("   ✅ Optimizer state restored")
            else:
                logger.warning("   ⚠️  No optimizer state found - will start with fresh optimizer")
        
        return start_epoch, resume_train_steps, last_eval_loss, True
        
    except Exception as e:
        if rank == 0 and logger:
            logger.error(f"❌ Failed to load checkpoint {checkpoint_path}: {str(e)}")
            logger.info("🚀 Starting training from scratch instead")
        return 0, 0, None, False

def print_optimizer_params(optimizer, model, learning_rate, fine_tune_lr, logger):
    param_lrs = {}
    for pg in optimizer.param_groups:
        lr = pg.get('lr', None)
        for p in pg['params']:
            param_lrs[id(p)] = lr

    # Log header
    logger.info(f"{'Name':<60}  {'Req_grad':<10}  {'LR':<8}  {'Group'}")
    logger.info("-" * 90)

    for name, p in model.named_parameters():
        lr = param_lrs.get(id(p), None)
        # decide group label
        if lr == learning_rate:
            grp = "New"
        elif lr == fine_tune_lr:
            grp = "Fine"
        else:
            grp = ""
        logger.info(f"{name:<60}  {str(p.requires_grad):<10}  {str(lr):<8}  {grp}")

def debugs_for_optimizer(model_module, config, debug, pretrained_flag, new_param_names, pretrained_params):
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model_module.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_module.parameters())
    
    # Always show basic optimizer info
    print("\nOptimizer Configuration:")
    print(f"- Training mode: {'FROM SCRATCH' if not pretrained_flag else 'FINE-TUNING'}")
    print(f"- Model depth: {getattr(model_module, 'depth', 'Not available')}")
    print(f"- Learning rate: {config.training.learning_rate}")
    print(f"- Total trainable parameters: {trainable_params:,}")
    print(f"- Total parameters: {total_params:,}")
    
    # Only show detailed info in debug mode
    if debug:
        if pretrained_flag:
            trainable_new_params = sum(p.numel() for p in new_params)
            trainable_pretrained_params = sum(p.numel() for p in pretrained_params)
            fine_tune_lr = getattr(config.training, 'fine_tune_lr', config.training.learning_rate * 0.1)
            print(f"- New parameters learning rate: {config.training.learning_rate}")
            print(f"- Pretrained parameters learning rate: {fine_tune_lr}")
            print(f"- Trainable new parameters: {trainable_new_params:,}")
            print(f"- Trainable pretrained parameters: {trainable_pretrained_params:,}")
        
        # Print status of each module in the model
        print("\nModule Status:")
        module_status = {}
        
        # Collect module information
        for name, param in model_module.named_parameters():
            # Extract module name (first part of parameter name)
            module_name = name.split('.')[0] if '.' in name else name
            
            # Initialize module info if not exists
            if module_name not in module_status:
                module_status[module_name] = {
                    'trainable_params': 0,
                    'frozen_params': 0,
                    'new_params': 0,
                    'pretrained_params': 0,
                    'total_params': 0
                }
            
            # Count parameters
            param_count = param.numel()
            module_status[module_name]['total_params'] += param_count
            
            if param.requires_grad:
                module_status[module_name]['trainable_params'] += param_count
                if pretrained_flag and name in new_param_names:
                    module_status[module_name]['new_params'] += param_count
                else:
                    module_status[module_name]['pretrained_params'] += param_count
            else:
                module_status[module_name]['frozen_params'] += param_count
        
        # Print module status
        for module_name, stats in sorted(module_status.items()):
            trainable = stats['trainable_params']
            frozen = stats['frozen_params']
            total = stats['total_params']
            new_params = stats['new_params']
            pretrained_params = stats['pretrained_params']
            
            train_percent = (trainable / total * 100) if total > 0 else 0
            
            print(f"- {module_name}:")
            print(f"  - Status: {'Trainable' if trainable > 0 else 'Frozen'}")
            print(f"  - Trainable parameters: {trainable:,} ({train_percent:.2f}%)")
            
            if trainable > 0:
                if pretrained_flag:
                    if new_params > 0:
                        print(f"    - New parameters: {new_params:,} (LR: {config.training.learning_rate})")
                    if pretrained_params > 0:
                        fine_tune_lr = getattr(config.training, 'fine_tune_lr', config.training.learning_rate * 0.1)
                        print(f"    - Pretrained parameters: {pretrained_params:,} (LR: {fine_tune_lr})")
                else:
                    print(f"    - Learning rate: {config.training.learning_rate}")
            
            print(f"  - Frozen parameters: {frozen:,}")
            print(f"  - Total parameters: {total:,}")

def log_params(model, logger):
    # Check if blocks attribute exists (handles depth=0 models)
    if hasattr(model.module, 'blocks'):
        logger.info("Logging parameters from transformer blocks:")
        for name, param in model.module.blocks.named_parameters():
            if param.numel() < 100:
                logger.info(f"{name} | Shape: {param.shape} | Values: {param.tolist()}")
            else:
                logger.info(f"{name} | Shape: {param.shape} | Min: {param.min().item():.4f} | Max: {param.max().item():.4f} | Mean: {param.mean().item():.4f}")
    else:
        # For models without blocks (depth=0), log all parameters
        logger.info("Model has no transformer blocks (depth=0). Logging all parameters:")
        for name, param in model.module.named_parameters():
            if param.numel() < 100:
                logger.info(f"{name} | Shape: {param.shape} | Values: {param.tolist()}")
            else:
                logger.info(f"{name} | Shape: {param.shape} | Min: {param.min().item():.4f} | Max: {param.max().item():.4f} | Mean: {param.mean().item():.4f}")

class Args:
    def __init__(self, config):
        self.image_size = config.data.image_size
        self.data_path = config.data.path
        self.results_dir = config.output.results_dir
        self.model = config.model.name
        self.num_classes = config.model.num_classes
        self.epochs = config.training.epochs
        self.global_batch_size = config.training.batch_size
        self.global_seed = config.training.seed
        self.num_workers = config.data.num_workers
        self.lr = config.training.learning_rate
        self.log_every = config.logging.log_every
        self.ckpt_every = config.logging.ckpt_every
        self.eval_every = config.logging.eval_every
        self.wandb_project = config.wandb.project
        self.wandb_entity = config.wandb.entity
        self.gradient_clip = config.training.gradient_clip


def initialize_model_with_pretrained(model, config, device, gain=0.3, rank=0, logger=None):
    model_to_init = model.module if hasattr(model, "module") else model
    new_param_names = set()
    pretrained_flag = False

    # Check if this is a DiT model with depth > 0
    is_dit_model = hasattr(model_to_init, 'depth')
    is_depth_gt_0 = is_dit_model and model_to_init.depth > 0

    if getattr(config.model, 'pretrained_path', None):
        pretrained_flag = True
        # Load pretrained weights
        pretrained_dict = torch.load(config.model.pretrained_path, map_location=device)
        
        # Extract model or EMA weights from checkpoint
        if 'model' in pretrained_dict:
            pretrained_dict = pretrained_dict['model']
        elif 'ema' in pretrained_dict:
            pretrained_dict = pretrained_dict['ema']
            if rank == 0:
                print("Using EMA weights from pretrained checkpoint")
        
        # Remove 'module.' prefix from pretrained keys if present
        pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
        
        # Get current model state dict
        model_state_dict = model_to_init.state_dict()
        
        # Identify new parameters (those not in pretrained weights)
        for name in model_state_dict.keys():
            # Add t_embedder.fine_head parameters to new_param_names for training
            if name.startswith('t_embedder.fine_head.') or name not in pretrained_dict:
                new_param_names.add(name)
            elif model_state_dict[name].shape != pretrained_dict[name].shape:
                new_param_names.add(name)  # Also consider shape-mismatched parameters as new
        
        if rank == 0:
            print("\n=== Parameter Analysis ===")
            print(f"Total parameters in model: {len(model_state_dict)}")
            print(f"New parameters identified: {len(new_param_names)}")
            print("\nNew parameters:")
            for name in sorted(new_param_names):
                print(f"- {name}")
            
            if is_dit_model:
                print(f"\nDiT model detected with depth={model_to_init.depth}")
            
            if is_depth_gt_0:
                print("This is a DiT model with depth > 0. The MLP path will be frozen.")
        
        # Initialize all weights first
        if hasattr(model_to_init, 'initialize_weights'):
            model_to_init.initialize_weights(gain=gain)
        
        # Now load the pretrained weights
        filtered_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_state_dict and k not in new_param_names:
                if v.shape == model_state_dict[k].shape:
                    filtered_dict[k] = v
        
        # Load filtered weights
        model_to_init.load_state_dict(filtered_dict, strict=False)
        
        # Special handling for DiT models with depth > 0
        if is_depth_gt_0:
            # 1. Freeze the MLP path (mlp_denoise)
            if hasattr(model_to_init, 'coarse'):
                requires_grad(model_to_init.coarse, False)
                if rank == 0:
                    print("Frozen MLP denoiser weights")
            
            # 2. Freeze the shared MLP and coarse head in timestep embedder
            if hasattr(model_to_init, 't_embedder'):
                requires_grad(model_to_init.t_embedder.mlp, False)
                requires_grad(model_to_init.t_embedder.coarse_head, False)
                # Ensure fine_head is trainable
                if hasattr(model_to_init.t_embedder, 'fine_head'):
                    requires_grad(model_to_init.t_embedder.fine_head, True)
                if rank == 0:
                    print("Frozen shared MLP and coarse head in timestep embedder")
                    print("Fine head in timestep embedder set as trainable")
            
            # 3. Ensure all other parameters are trainable
            for name, param in model_to_init.named_parameters():
                if not name.startswith('coarse.') and not name.startswith('t_embedder.mlp.') and not name.startswith('t_embedder.coarse_head.'):
                    param.requires_grad = True
        else:
            # For all other models, keep all parameters trainable
            for name, param in model_to_init.named_parameters():
                param.requires_grad = True
        
        if rank == 0:
            print("\n=== Weight Loading Summary ===")
            print(f"Successfully loaded: {len(filtered_dict)} / {len(model_state_dict)} parameters")
            print("\nParameter status:")
            trainable_params = sum(p.numel() for p in model_to_init.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model_to_init.parameters())
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Total parameters: {total_params:,}")
            
            if is_depth_gt_0:
                print("\nTraining configuration for DiT with depth > 0:")
                print("- MLP path: FROZEN (using pretrained weights)")
                print("- Shared MLP in timestep embedder: FROZEN")
                print("- Coarse head in timestep embedder: FROZEN")
                print("- Fine head in timestep embedder: TRAINABLE (initialized as new)")
                print("- Transformer path: TRAINABLE")

    return model, pretrained_flag, new_param_names


MAX_CHECKPOINTS = 3  # Keep only the newest 3 checkpoints

def manage_checkpoints(checkpoint_dir: str, rank: int) -> None:
    if rank != 0:
        return
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_files = sorted(glob(os.path.join(checkpoint_dir, "*.pt")), key=os.path.getmtime)
    if len(checkpoint_files) > MAX_CHECKPOINTS:
        # Remove oldest files
        for old_file in checkpoint_files[:-MAX_CHECKPOINTS]:
            try:
                os.remove(old_file)
            except OSError:
                pass


def weights_detection(model, device, pretrained_path):
    try:
        # Load the state dict
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
            
        pretrained_dict = torch.load(pretrained_path)
        print(f"\nLoading pretrained weights from: {pretrained_path}")
        
        # Handle both cases where pretrained_dict is a state_dict or contains it
        if 'model' in pretrained_dict:
            pretrained_dict = pretrained_dict['model']
            print("Found 'model' key in pretrained dict")
            
        # Filter and update weights with better error handling
        filtered_dict = {}
        matched_params = []
        mismatched_params = []
        missing_params = []
        
        # First, check which parameters are available
        for k in model_state_dict.keys():
            if k not in pretrained_dict:
                missing_params.append(k)
                
        # Then process the pretrained weights
        for k, v in pretrained_dict.items():
            # Remove 'module.' prefix if it exists
            k = k.replace('module.', '')
            if k in model_state_dict:
                if v.shape == model_state_dict[k].shape:
                    filtered_dict[k] = v
                    matched_params.append(f"{k} (shape: {v.shape})")
                else:
                    mismatched_params.append(
                        f"{k} (pretrained: {v.shape}, model: {model_state_dict[k].shape})"
                    )
        
        # Print detailed loading information
        print("\n=== Weight Loading Summary ===")
        print(f"\nSuccessfully matched parameters ({len(matched_params)}):")
        for param in matched_params:
            print(f"✓ {param}")
            
        if mismatched_params:
            print(f"\nShape mismatched parameters ({len(mismatched_params)}):")
            for param in mismatched_params:
                print(f"× {param}")
                
        if missing_params:
            print(f"\nMissing parameters ({len(missing_params)}):")
            for param in missing_params:
                print(f"? {param}")

        # Update model weights
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(filtered_dict, strict=False)
        else:
            model.load_state_dict(filtered_dict, strict=False)

        print(f"\nTotal parameters loaded: {len(filtered_dict)}/{len(model_state_dict)}")
        return model
        
    except Exception as e:
        print(f"Error loading weights: {e}")
        raise

def convert_to_numeric(value: Any) -> Any:
    if isinstance(value, str):
        try:
            if value.isdigit():
                return int(value)
            return float(value)
        except ValueError:
            return value
    elif isinstance(value, dict):
        return {k: convert_to_numeric(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [convert_to_numeric(item) for item in value]
    return value

class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if config is None:
        return Config({})
    converted_config = convert_to_numeric(config)
    return Config(converted_config)

def set_random_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    from monai.utils import set_determinism
    set_determinism(seed=seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_training_environment(args: Any) -> tuple:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
    
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    set_random_seed(seed)
    
    if rank == 0:
        experiment_dir, checkpoint_dir = create_experiment_dirs(args)
        logger = create_logger(experiment_dir)
        logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}")
    else:
        experiment_dir = checkpoint_dir = None
        logger = create_logger(None)
        
    return rank, device, experiment_dir, checkpoint_dir, logger

def create_experiment_dirs(args: Any) -> Tuple[str, str]:
    os.makedirs(args.results_dir, exist_ok=True)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    return experiment_dir, checkpoint_dir

def setup_dataloader(args: Any,
                     rank: int = 0,
                     config: Config = None,
                     data_type: str = "train",
                     cache_dir: str = None,
                     val_frac: float = 0.1,
                     test_frac: float = 0.1,
                     roi_size: Tuple[int, int, int] = (32, 32, 32),
                     preprocess: str = None,
                     normalize: bool = False,
                     seed: int = 42,
                     batch_size: int = None,
                     augment: bool = True) -> DataLoader:
    dataset = get_voxel_dataset(
        args.data_path,
        config=config,
        task=args.task,
        roi_size=roi_size,
        data_type=data_type,
        val_frac=val_frac,
        test_frac=test_frac,
        preprocess=preprocess,
        normalize=normalize,
        seed=seed,
        augment=augment,
        cache_dir=cache_dir,
        rank=rank,
    )

    # Smart: Only use distributed sampler if running on multiple GPUs
    is_distributed = (
        dist.is_available() and
        dist.is_initialized() and
        dist.get_world_size() > 1
    )

    should_shuffle = data_type == "train"
    if rank == 0:
        print(f"Data type: {data_type}, Shuffle: {should_shuffle}")

    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=should_shuffle,
            seed=args.global_seed
        )
        if batch_size is None:
            batch_size = int(args.global_batch_size // dist.get_world_size())
        else:
            batch_size = batch_size
        shuffle = False  # Don't use DataLoader's shuffle when using sampler
        if rank == 0:
            print(f"Distributed training with batch size: {batch_size}, world size: {dist.get_world_size()}")
    else:
        sampler = None
        if batch_size is None:
            batch_size = args.global_batch_size
        else:
            batch_size = batch_size
        shuffle = should_shuffle  # Use DataLoader shuffle directly
        if rank == 0:
            print(f"Single GPU training with batch size: {batch_size}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

def save_depth_model_visualizations(
    samples: torch.Tensor,
    coarse_out: torch.Tensor,
    fine_out: torch.Tensor,
    experiment_dir: str,
    image_size: int,
    epoch: int,
    rank: int,
    nii_number: Optional[int] = 2,
    logger: logging.Logger = None
) -> None:
    # Split predictions into noise and image components
    coarse_noise, coarse_img = coarse_out.chunk(2, dim=1)
    fine_noise, fine_img = fine_out.chunk(2, dim=1)
    
    # Create directories for visualization
    coarse_dir = f"{experiment_dir}/coarse_path"
    fine_dir = f"{experiment_dir}/fine_path"
    coarse_noise_dir = f"{experiment_dir}/coarse_path_noise"
    fine_noise_dir = f"{experiment_dir}/fine_path_noise"
    os.makedirs(coarse_dir, exist_ok=True)
    os.makedirs(fine_dir, exist_ok=True)
    os.makedirs(coarse_noise_dir, exist_ok=True)
    os.makedirs(fine_noise_dir, exist_ok=True)
    
    if rank == 0:
        # Save visualizations for both image and noise predictions
        save_evaluation_samples(
            samples, f"{experiment_dir}/noisy", image_size, epoch, nii_number, logger
        )
        # Save image predictions
        save_evaluation_samples(
            coarse_img, coarse_dir, image_size, epoch, nii_number, logger
        )   
        save_evaluation_samples(
            fine_img, fine_dir, image_size, epoch, nii_number, logger
        )
        # Save noise predictions
        save_evaluation_samples(
            coarse_noise, coarse_noise_dir, image_size, epoch, nii_number, logger
        )
        save_evaluation_samples(
            fine_noise, fine_noise_dir, image_size, epoch, nii_number, logger
        )
        logger.info(f"Saved visualization of both coarse and fine path outputs (image and noise predictions)")

def save_evaluation_samples(
    samples: torch.Tensor,
    experiment_dir: str,
    image_size: int,
    epoch: int,
    nii_number: Optional[int] = None,
    logger: logging.Logger = None
) -> None:
    import matplotlib.pyplot as plt
    
    def save_orthogonal_views(sample: torch.Tensor, idx: int) -> None:
        # Get middle slices along each axis
        mid_slice = image_size // 2
        slices = [
            sample[:, :, mid_slice],  # sagittal
            sample[:, mid_slice, :],  # coronal
            sample[mid_slice, :, :]   # axial
        ]
        slices = [s.cpu().float().numpy() for s in slices]
        
        # Create and save figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plt.subplots_adjust(wspace=0, hspace=0)
        
        for ax, slice_data in zip(axes, slices):
            ax.imshow(slice_data, cmap='gray')
            ax.axis('off')
            
        plt.savefig(
            f"{vis_dir}/sample_{idx}_views_epoch_{epoch}.png",
            bbox_inches='tight',
            pad_inches=0,
            dpi=300
        )
        plt.close(fig)

    def save_nifti_volumes(samples_to_save: torch.Tensor) -> None:
        try:
            import nibabel as nib
            for i, sample in enumerate(samples_to_save):
                nib.save(
                    nib.Nifti1Image(
                        sample.squeeze(0).cpu().float().numpy(),
                        np.eye(4)
                    ),
                    f"{experiment_dir}/sample_{i}_{epoch}.nii.gz"
                )
            if logger:
                logger.info(f"Saved {len(samples_to_save)} NIfTI volumes at epoch {epoch}")
        except ImportError:
            if logger:
                logger.warning("nibabel not installed, skipping NIfTI saving")

    # Create necessary directories
    os.makedirs(experiment_dir, exist_ok=True)
    vis_dir = os.path.join(experiment_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # Process samples
    batch_size = samples.shape[0]
    samples = samples.detach()  # Ensure we're not tracking gradients

    # Save orthogonal views for all samples
    for idx in range(batch_size):
        save_orthogonal_views(samples[idx, 0], idx)
    
    if logger:
        logger.info(f"Saved orthogonal views visualization for {batch_size} samples at epoch {epoch}")

    # Save NIfTI volumes
    if nii_number is not None:
        samples_to_save = samples[:min(nii_number, batch_size)]
    else:
        samples_to_save = samples
    
    save_nifti_volumes(samples_to_save)

@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.9999) -> None:
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir: Optional[str]) -> logging.Logger:
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def check_background_dominance_raw(arr1, arr2,
                                   bg_threshold=5.0):
    assert isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray)
    assert arr1.shape == arr2.shape
    assert arr1.ndim == 4

    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)

    se = (arr1 - arr2) ** 2

    bg_mask = arr1 < bg_threshold
    fg_mask = ~bg_mask

    if not bg_mask.any():
        print("No background voxels under threshold; try a higher bg_threshold.")
        return
    if not fg_mask.any():
        print("No foreground voxels; threshold too high.")
        return

    se_total = se.sum()
    se_bg = se[bg_mask].sum()
    se_fg = se[fg_mask].sum()

    bg_ratio = bg_mask.mean()
    fg_ratio = fg_mask.mean()

    mse_total = se_total / se.size
    mse_bg = se_bg / bg_mask.sum()
    mse_fg = se_fg / fg_mask.sum()

    frac_bg = se_bg / se_total
    frac_fg = se_fg / se_total

    print("==== Background dominance check (raw range) ====")
    print(f"bg_threshold = {bg_threshold}")
    print(f"bg_ratio (voxel fraction) = {bg_ratio:.4f}")
    print(f"fg_ratio (voxel fraction) = {fg_ratio:.4f}")
    print(f"Total MSE        = {mse_total:.3f}")
    print(f"MSE (background) = {mse_bg:.3f}")
    print(f"MSE (foreground) = {mse_fg:.3f}")
    print(f"SE fraction (BG) = {frac_bg:.4f}")
    print(f"SE fraction (FG) = {frac_fg:.4f}")
    print("===============================================")

    return {
        "bg_ratio": bg_ratio,
        "fg_ratio": fg_ratio,
        "mse_total": mse_total,
        "mse_bg": mse_bg,
        "mse_fg": mse_fg,
        "frac_bg": frac_bg,
        "frac_fg": frac_fg,
    }
