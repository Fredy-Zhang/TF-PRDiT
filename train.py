# ============================================================================
# IMPORTS
# ============================================================================

# Standard library imports
import argparse
import logging
import os
import random
from copy import deepcopy
from glob import glob
from time import time, sleep
from typing import Any, Dict, Optional, Tuple

# from collections import deque
# from datetime import datetime
# from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb
from tqdm import tqdm

# Local imports
from diffusion import loading_diffusion
from models import load_model
from util import (
    Args, Config, create_experiment_dirs, create_logger,
    debugs_for_optimizer, evaluate_model_depth_zero,
    find_latest_checkpoint, initialize_model_with_pretrained,
    load_checkpoint_state, load_config, log_params,
    print_optimizer_params, requires_grad, sample_from_model,
    save_evaluation_samples, setup_dataloader, setup_torch_config,
    setup_wandb, update_ema
)


def optimize_ddp_model(model: torch.nn.Module, find_unused_parameters: bool = False) -> torch.nn.Module:
    """Wrap a model in DDP with the communication settings used by this project."""
    if not dist.is_initialized():
        return model
    device_id = dist.get_rank() % torch.cuda.device_count()
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device_id],
        output_device=device_id,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=True,
        static_graph=True,
        bucket_cap_mb=100,
    )


# ============================================================================
# CONSTANTS
# ============================================================================

DEBUG = False


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
# Functions for setting up train and validation data loaders

def return_train_val_loaders(args, rank, config, seed: int, debug: bool = False):
    """Setup train and validation data loaders."""
    args.task = config.data.task
    args.data_path = config.data.path
    
    loader = setup_dataloader(
        args=args,
        rank=rank,
        config=config,
        data_type="train",
        roi_size=(config.data.image_size,) * 3,
        augment=config.data.augment,
        normalize=config.data.normalize,
        seed=seed,
    )
    
    val_loader = setup_dataloader(
        args=args,
        rank=rank,
        config=config,
        data_type="val",
        roi_size=(config.data.image_size,) * 3,
        augment=False,
        normalize=config.data.normalize,
        seed=config.training.seed,
        batch_size=4,
    )
    
    return loader, val_loader


# ============================================================================
# MODEL SETUP FUNCTIONS
# ============================================================================
# Functions for initializing and configuring models and optimizers

def initialize_optimizer(model, config, pretrained_flag, new_param_names, rank=0, debug=False):
    """Initialize AdamW optimizer with different learning rates for pretrained and new parameters."""
    model_module = model.module if hasattr(model, "module") else model
    
    param_groups = []
    new_params = []
    pretrained_params = []
    
    if not pretrained_flag:
        param_groups.append({
            'params': [p for p in model_module.parameters() if p.requires_grad],
            'lr': config.training.learning_rate,
            'weight_decay': 0
        })
    else:
        for name, param in model_module.named_parameters():
            if param.requires_grad:
                if name in new_param_names:
                    new_params.append(param)
                else:
                    pretrained_params.append(param)
        
        if new_params:
            param_groups.append({
                'params': new_params,
                'lr': config.training.learning_rate,
                'weight_decay': 0
            })
        
        if pretrained_params:
            fine_tune_lr = getattr(config.training, 'fine_tune_lr', config.training.learning_rate * 0.1)
            param_groups.append({
                'params': pretrained_params,
                'lr': fine_tune_lr,
                'weight_decay': 0
            })
    
    optimizer = torch.optim.AdamW(param_groups)
    
    if rank == 0:
        debugs_for_optimizer(model_module, config, debug, pretrained_flag, new_param_names, pretrained_params)

    return optimizer


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
# Core training step and loss computation functions

# Flag to track if data range has been logged (only log once)
_has_logged_data_range = False

def train_step(model, diffusion, x, optimizer, ema, device, 
               gradient_clip: float,
               conditions: Optional[torch.Tensor] = None,
               current_epoch: Optional[int] = None) -> Optional[Tuple[float, float, float, Optional[float]]]:
    """
    Optimized single training step with better error handling and performance.
    
    Returns:
        Tuple of (total_loss, noise_loss, img_loss, ssim_loss)
        ssim_loss will be None if not computed (before epoch 6000)
    """
    if isinstance(device, int):
        device = torch.device(f'cuda:{device}')
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Log data range of input x (only once)
    global _has_logged_data_range
    if not _has_logged_data_range:
        x_min = x.min().item()
        x_max = x.max().item()
        x_mean = x.mean().item()
        x_std = x.std().item()
        logging.info("="*30)
        logging.info(f"Input data range - min: {x_min:.6f}, max: {x_max:.6f}, mean: {x_mean:.6f}, std: {x_std:.6f}")
        logging.info("="*30)
        _has_logged_data_range = True
    
    t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
    loss_dict = diffusion.training_losses(model, x, t, conditions=conditions, current_epoch=current_epoch)
    
    if isinstance(loss_dict, dict) and "noise_loss" in loss_dict and "img_loss" in loss_dict:
        noise_loss = loss_dict["noise_loss"].mean()
        img_loss = loss_dict["img_loss"].mean()
        total_loss = noise_loss + img_loss
        ssim_loss = None  # SSIM loss disabled
        
    else:
        total_loss = loss_dict["loss"].mean() if isinstance(loss_dict, dict) else loss_dict.mean()
        noise_loss = img_loss = total_loss
        ssim_loss = None
    
    if not torch.isfinite(total_loss) or total_loss > 1e5:
        return None, None, None, None
    
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    
    model_module = model.module if hasattr(model, 'module') else model
    is_depth_zero = hasattr(model_module, 'depth') and model_module.depth == 0
    
    if is_depth_zero:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if trainable_params:
            torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip)
    
    optimizer.step()
    
    update_ema(ema, model_module)
    
    ssim_loss_item = ssim_loss.item() if ssim_loss is not None else None
    return total_loss.item(), noise_loss.item(), img_loss.item(), ssim_loss_item


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================
# Functions for model evaluation and sample generation

def evaluate_model(
    model: torch.nn.Module,
    val_loader,
    diffusion: Any,
    device: torch.device,
    rank: int,
    experiment_dir: str,
    image_size: int,
    epoch: int,
    logger: logging.Logger,
    train_steps: int = None,
    config: Any = None
) -> Dict[str, float]:
    """Evaluates model and saves samples."""
    if isinstance(device, int):
        device = torch.device(f'cuda:{device}')
    elif isinstance(device, str):
        device = torch.device(device)

    inner_model = model.module if hasattr(model, 'module') else model
    is_depth_zero = hasattr(inner_model, 'depth') and inner_model.depth == 0

    if not is_depth_zero:
        sample_from_model(
            ema_model=model,
            diffusion=diffusion,
            device=device,
            rank=rank,
            experiment_dir=experiment_dir,
            image_size=image_size,
            epoch=epoch,
            logger=logger,
            num_samples=4,
            batch_size=2,
            train_steps=train_steps,
            config=config
        )
    else:
        evaluate_model_depth_zero(
            model=model,
            val_loader=val_loader,
            diffusion=diffusion,
            device=device,
            rank=rank,
            experiment_dir=experiment_dir,
            image_size=image_size,
            epoch=epoch,
            logger=logger,
            train_steps=train_steps,
        )


# ============================================================================
# TRAINER CLASS
# ============================================================================
# Main training class that orchestrates the entire training process

class Trainer:
    """Optimized trainer with best loss checkpoint management and efficient distributed training."""
    
    def __init__(self, config: Config, rank: int, device: torch.device, seed: int, resume_training: bool = False, debug: bool = False, from_scratch: bool = False):
        self.config = config
        self.from_scratch = from_scratch
        self.rank = rank
        self.device = device
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.is_distributed = self.world_size > 1
        
        self.resume_training = resume_training
        self.start_epoch = 0
        self.resume_train_steps = 0
        self.is_resumed = False
        self.debug = debug
        
        if rank == 0:
            self.experiment_dir, self.checkpoint_dir = create_experiment_dirs(Args(config))
        else:
            self.experiment_dir = self.checkpoint_dir = None
            
        self.logger = create_logger(None if rank != 0 else self.experiment_dir)
            
        if self.config.wandb.enable:
            self._setup_wandb()
            
        self._setup_model()
        self._setup_data(seed=seed)
        self._check_resume_training()
    
    # ========================================================================
    # SETUP METHODS
    # ========================================================================
    # Methods for initializing various components
    
    def _setup_wandb(self):
        """Initialize wandb tracking"""
        if self.rank == 0:
            setup_wandb(self.config, self.rank)
            
    def _setup_model(self):
        """Initialize model, EMA, optimizer and diffusion"""
        self.model = load_model(self.config)
        
        if hasattr(self.model, 'log_config'):
            self.model.log_config(self.rank)
        
        is_depth_zero = hasattr(self.model, 'depth') and self.model.depth == 0
        if is_depth_zero:
            self.model.t_embedder.fine_head.requires_grad_(False)
        
        self.model = self.model.to(self.device)
        
        self.ema = deepcopy(self.model).to(self.device)
        requires_grad(self.ema, False)
        
        if self.world_size > 1:
            self.model = optimize_ddp_model(self.model, find_unused_parameters=False)
        
        self.diffusion = loading_diffusion(self.config, self.rank)
        
        if self.rank == 0 and self.debug:
            self.logger.info("\n=== Before state of all block parameters ===")
            log_params(self.model, self.logger)
        
        pretrained_flag = False
        if self.from_scratch:
            new_param_names = None
        else:
            pretrained_path = getattr(self.config.model, 'pretrained_path', None)
            if pretrained_path and self.rank == 0:
                self.logger.info("="*60)
                self.logger.info("Loading pretrained weights")
                self.logger.info("="*60)
                self.logger.info(f"📥 Loading pretrained weights from: {pretrained_path}")
                self.logger.info("*"*60)
                for _ in tqdm(range(10), desc="Waiting", unit="s", ncols=80):
                    sleep(1)
            self.model, pretrained_flag, new_param_names = initialize_model_with_pretrained(
                model=self.model,
                config=self.config,
                device=f"cuda:{self.rank}",
                rank=self.rank,
                gain=0.1
            )
        
        self.optimizer = initialize_optimizer(
            self.model, self.config, pretrained_flag=pretrained_flag, 
            new_param_names=new_param_names, rank=self.rank, debug=self.debug
        )
        
        if self.rank == 0 and self.debug:
            fine_tune_lr = getattr(self.config.training, 'fine_tune_lr', self.config.training.learning_rate * 0.1)
            print_optimizer_params(self.optimizer, self.model, 
                                   self.config.training.learning_rate, 
                                   fine_tune_lr, self.logger)
        if self.rank == 0 and self.debug:
            self.logger.info("\n=== Final state of all block parameters ===")
            log_params(self.model, self.logger)
    
    def _setup_data(self, seed: int):
        """Initialize data loaders"""
        args = Args(self.config)
        if self.rank == 0 and self.debug:
            print(f"Setting up data loaders with seed: {seed}")
        self.train_loader, self.val_loader = return_train_val_loaders(
            args, self.rank, self.config, seed, self.debug
        )
        
        if self.rank == 0:
            if self.debug:
                self.logger.info(f"DiT Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            self.logger.info(f"Dataset contains {len(self.train_loader.dataset):,} images ({self.config.data.path})")
            
    def _setup_training_state(self):
        """Initialize training state"""
        model_for_ema = self.model.module if hasattr(self.model, 'module') else self.model
        update_ema(self.ema, model_for_ema, decay=0)
        self.model.train()
        self.ema.eval()
        
        model_inner = self.model.module if hasattr(self.model, 'module') else self.model
        is_depth_zero = hasattr(model_inner, 'depth') and model_inner.depth == 0
        
        if self.config.model.flash_attn and not is_depth_zero:
            if self.rank == 0 and self.debug:
                self.logger.info("Enabling flash attention optimizations for transformer model")
            torch.set_float32_matmul_precision('high')
        else:
            if is_depth_zero and self.rank == 0 and self.debug:
                self.logger.info("Using full precision for depth=0 model (no attention)")
        
        self.model = self.model.to(dtype=torch.float32)
        self.ema = self.ema.to(dtype=torch.float32)
    
    # ========================================================================
    # CHECKPOINT METHODS
    # ========================================================================
    # Methods for saving, loading, and managing checkpoints
    
    def save_checkpoint(self, epoch: int, train_steps: int, loss: float = None, save_optimizer: bool = False):
        """Save checkpoint to disk with 6-digit epoch format."""
        if self.rank != 0:
            return
            
        checkpoint_data = {
            "model": self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "config": self.config.to_dict(),
            "epoch": epoch,
            "train_steps": train_steps,
            "loss": loss
        }
        
        if save_optimizer:
            checkpoint_data["optimizer"] = self.optimizer.state_dict()
        
        filename = f"{epoch:06d}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(checkpoint_data, checkpoint_path)
        
        size_info = " (with optimizer)" if save_optimizer else " (model + EMA only)"
        self.logger.info(f"💾 Saved checkpoint: {checkpoint_path}{size_info}")
    
    def _check_resume_training(self):
        """Check if resume training is enabled and load latest checkpoint."""
        if not self.resume_training:
            if self.rank == 0:
                self.logger.info("🚀 Starting training from scratch")
            return
        
        if self.rank == 0:
            latest_checkpoint = find_latest_checkpoint(self.checkpoint_dir)
            if latest_checkpoint:
                self.logger.info(f"🔄 Found checkpoint for resume: {latest_checkpoint}")
                self._load_checkpoint(latest_checkpoint)
            else:
                self.logger.info("⚠️  Resume training enabled but no checkpoints found. Starting from scratch.")
        
        if self.is_distributed:
            resume_data = torch.tensor([self.start_epoch, self.resume_train_steps, int(self.is_resumed)], 
                                     dtype=torch.long, device=self.device)
            dist.broadcast(resume_data, src=0)
            
            if self.rank != 0:
                self.start_epoch = int(resume_data[0].item())
                self.resume_train_steps = int(resume_data[1].item())
                self.is_resumed = bool(resume_data[2].item())
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and restore training state."""
        self.start_epoch, self.resume_train_steps, last_eval_loss, success = load_checkpoint_state(
            checkpoint_path, self.model, self.ema, self.optimizer, 
            self.device, self.rank, self.logger
        )
        self.is_resumed = success
        
        if last_eval_loss is not None:
            self._last_eval_loss = last_eval_loss
    
    def _cleanup_old_checkpoints(self):
        """Keep only the most recent checkpoints to save disk space."""
        if self.rank != 0:
            return
            
        checkpoint_files = sorted(glob(os.path.join(self.checkpoint_dir, "*.pt")))
        if len(checkpoint_files) > 5:  
            for old_file in checkpoint_files[:-5]:
                try:
                    os.remove(old_file)
                    self.logger.debug(f"Removed old checkpoint: {old_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove {old_file}: {e}")
    
    # ========================================================================
    # TRAINING METHODS
    # ========================================================================
    # Methods for the main training loop
    
    def train(self):
        """Main training loop"""
        self._setup_training_state()
        
        if self.is_distributed:
            torch.cuda.synchronize()
            dist.barrier()
            
        train_steps = self.resume_train_steps
        log_steps = running_loss = running_noise_loss = running_img_loss = running_ssim_loss = 0
        start_time = time()
        
        if self.rank == 0:
            if self.is_resumed:
                remaining_epochs = self.config.training.epochs - self.start_epoch
                print(f"🔄 Resuming training on {self.world_size} GPU(s) from epoch {self.start_epoch}...")
                self.logger.info(f"Resumed from epoch {self.start_epoch}, training for {remaining_epochs} more epochs to reach {self.config.training.epochs}")
            else:
                print(f"🚀 Starting training on {self.world_size} GPU(s)...")
                self.logger.info(f"Training for {self.config.training.epochs} epochs...")
            
            if self.config.wandb.enable and wandb.run is not None:
                wandb.log({"train/epoch": self.start_epoch, "status": "started"}, step=0)
        
        for epoch in range(self.start_epoch, self.config.training.epochs):
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            if self.rank == 0 and epoch == self.start_epoch and self.is_resumed:
                progress = (epoch / self.config.training.epochs) * 100
                self.logger.info(f"📊 Resuming at epoch {epoch} ({progress:.1f}% complete)")

            for batch in self.train_loader:
                if DEBUG:
                    image = batch["image"]
                    save_evaluation_samples(samples=image, experiment_dir=self.experiment_dir, 
                                            image_size=128, epoch=epoch, nii_number=None)
                    continue
                if self.config.training.conditions:
                    conditions = batch["x_rays"].to(self.device, non_blocking=True)
                else:
                    conditions = None
                    
                total_loss, noise_loss, img_loss, ssim_loss = train_step(
                    self.model, 
                    self.diffusion, 
                    batch["image"].to(self.device, non_blocking=True), 
                    self.optimizer, 
                    self.ema, 
                    self.device, 
                    Args(self.config).gradient_clip,
                    conditions=conditions,
                    current_epoch=epoch
                )
                
                if total_loss is None:
                    if self.rank == 0:
                        self.logger.warning("⚠️  Skipping step: invalid loss detected")
                    continue
                
                running_loss += total_loss
                running_noise_loss += noise_loss
                running_img_loss += img_loss
                # SSIM loss tracking disabled
                # if ssim_loss is not None:
                #     running_ssim_loss += ssim_loss
                log_steps += 1
                train_steps += 1
                
                if train_steps % self.config.logging.log_every == 0 and log_steps > 0:
                    end_time = time()
                    steps_per_sec = log_steps / max(end_time - start_time, 1e-8)
                    
                    # Build stats tensor - include SSIM loss if available (DISABLED)
                    stats_list = [running_loss, running_noise_loss, running_img_loss]
                    # if running_ssim_loss > 0:
                    #     stats_list.append(running_ssim_loss)
                    
                    stats = torch.tensor(
                        stats_list,
                        device=self.device,
                        dtype=torch.float32,
                    ) / max(log_steps, 1)
                    
                    if dist.is_initialized() and dist.get_world_size() > 1:
                        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                        stats /= dist.get_world_size()
                    
                    stats_list = stats.tolist()
                    avg_loss, avg_noise_loss, avg_img_loss = stats_list[0], stats_list[1], stats_list[2]
                    avg_ssim_loss = None  # SSIM loss disabled
                    # avg_ssim_loss = stats_list[3] if len(stats_list) > 3 else None
                    
                    if self.rank == 0:
                        log_dict = {
                            "train/total_loss": avg_loss,
                            "train/noise_loss": avg_noise_loss, 
                            "train/img_loss": avg_img_loss,
                            "train/epoch": epoch,
                        }
                        # SSIM loss logging disabled
                        # if avg_ssim_loss is not None:
                        #     log_dict["train/ssim_loss"] = avg_ssim_loss
                        
                        if self.config.wandb.enable and wandb.run is not None:
                            wandb.log(log_dict)
                        
                        loss_str = f"Loss: {avg_loss:.6f} (noise: {avg_noise_loss:.6f}, img: {avg_img_loss:.6f})"
                        # SSIM loss logging disabled
                        # if avg_ssim_loss is not None:
                        #     loss_str += f", ssim: {avg_ssim_loss:.6f}"
                        loss_str += ")"
                        
                        self.logger.info(
                            f"📊 Epoch {epoch:4d} | Step {train_steps:6d} | "
                            f"{loss_str} | "
                            f"Speed: {steps_per_sec:.2f} steps/s"
                            f"{f' | Rank: {self.rank}/{self.world_size}' if self.is_distributed else ''}"
                        )
                    
                    running_loss = running_noise_loss = running_img_loss = running_ssim_loss = log_steps = 0  # SSIM loss tracking disabled
                    start_time = time()
            
            if (epoch + 1) % self.config.logging.eval_every == 0:
                self.evaluate(epoch, train_steps)
                self.logger.info(f"🎨 Epoch {epoch + 1:4d} | Generated samples.......\n")
            
            if (epoch + 1) % self.config.logging.ckpt_every == 0:
                if hasattr(self, '_last_eval_loss'):
                    current_loss = self._last_eval_loss
                else:
                    current_loss = None
                
                save_optimizer = ((epoch + 1) % 5000 == 0) or ((epoch + 1) == self.config.training.epochs)
                self.save_checkpoint(epoch + 1, train_steps, current_loss, save_optimizer)
                        
        if self.rank == 0:
            final_loss = getattr(self, '_last_eval_loss', None)
            self.save_checkpoint(self.config.training.epochs, train_steps, final_loss, save_optimizer=True)
            
            if self.is_resumed:
                self.logger.info(f"🏁 Resumed training completed! Started from epoch {self.start_epoch}, finished at epoch {self.config.training.epochs}")
            else:
                self.logger.info(f"🏁 Training completed from scratch! Total epochs: {self.config.training.epochs}")
            
            self.logger.info("🎉 Training completed successfully!")
            
            if self.config.wandb.enable and wandb.run is not None:
                wandb.finish()
    
    # ========================================================================
    # EVALUATION METHODS
    # ========================================================================
    # Methods for model evaluation
    
    def evaluate(self, epoch: int, train_steps: int = None) -> Optional[Dict[str, float]]:
        """Run evaluation, using depth-conditional path."""
        result = None
        if self.rank == 0:
            result = evaluate_model(
                model=self.ema,
                val_loader=self.val_loader,
                diffusion=self.diffusion,
                device=self.device,
                rank=self.rank,
                experiment_dir=self.experiment_dir,
                image_size=self.config.data.image_size,
                epoch=epoch,
                logger=self.logger,
                train_steps=train_steps,
                config=self.config,
            )
        
        if self.is_distributed:
            dist.barrier()
        
        return result
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get current training statistics from efficient trackers."""
        return {
            "status": "training_complete"
        }


# ============================================================================
# MAIN FUNCTION AND ENTRY POINT
# ============================================================================
# Functions for script execution and argument parsing

def main(config: Config, resume_training: bool = False, debug: bool = False, from_scratch: bool = False) -> None:
    """Optimized main training function with efficient distributed setup."""
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    stage_name = getattr(getattr(config, "stage", None), "name", "")
    if "stage2" in stage_name and not getattr(config.model, "pretrained_path", ""):
        raise ValueError("Stage 2 global residual training requires model.pretrained_path from stage 1.")
    
    setup_torch_config()
    
    dist.init_process_group("nccl")
    assert config.training.batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = config.training.seed * dist.get_world_size() + rank
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    
    trainer = Trainer(config, rank, device, seed, resume_training, debug, from_scratch=from_scratch)
    trainer.train()


def get_argument_parser() -> argparse.ArgumentParser:
    """Creates and returns the argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config YAML file name under configs/')
    parser.add_argument('--from_scratch', action='store_true', help='Train from scratch, only works with stage 1, depth=0')
    parser.add_argument('--resume_training', action='store_true', 
                       help='Resume training from the latest checkpoint found in checkpoint directory')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode with verbose logging (model structure, parameter details, etc.)')
    
    return parser


if __name__ == "__main__":
    parser = get_argument_parser()
    args = parser.parse_args()
    main(load_config(os.path.join("configs", args.config)), args.resume_training, args.debug, args.from_scratch)
"""
Stage1:
OMP_NUM_THREADS=4 torchrun --nnodes=1 --nproc_per_node=4 train.py --config lidc_stage1_local.yaml --from_scratch

OMP_NUM_THREADS=4 torchrun --nnodes=1 --nproc_per_node=4 train.py --config rad_chest_stage1_local.yaml --from_scratch

Stage2:
OMP_NUM_THREADS=4 torchrun --nnodes=1 --nproc_per_node=4 train.py --config lidc_stage2_global.yaml

OMP_NUM_THREADS=4 torchrun --nnodes=1 --nproc_per_node=4 train.py --config rad_chest_stage2_global.yaml
"""
