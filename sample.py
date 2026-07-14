# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import argparse
import os
import torch
from tqdm import tqdm

# Configure PyTorch settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from utils.download import find_model
from models import DiT_models
from util import load_config
from util import data_transform_backward
from diffusion.image_noise_diffusion import IaNDiffusion

def main():
    # Load configuration
    args = parse_args()
    
    # Validate arguments
    if not os.path.exists(os.path.join("configs", args.config)):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint file not found: {args.ckpt}")
    
    if args.num_samples <= 0 or args.total_samples <= 0:
        raise ValueError("num_samples and total_samples must be positive")
    
    config = load_config(os.path.join("configs", args.config))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup PyTorch:
    torch.manual_seed(config.training.seed)
    torch.set_grad_enabled(False)

    # Create output directories
    xs_path = os.path.join(args.output_dir, "xs")
    x0_path = os.path.join(args.output_dir, "x0")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(xs_path, exist_ok=True)
    os.makedirs(x0_path, exist_ok=True)
    
    # Load model
    try:
        if config.model.name not in DiT_models:
            raise ValueError(f"Unknown model: {config.model.name}. Available models: {list(DiT_models.keys())}")
            
        model = DiT_models[config.model.name](
            input_size=config.data.image_size,
            in_channels=config.model.in_channels,
            num_classes=config.model.num_classes,
            learn_sigma=True if config.model.out_channels == 2 else False,
            flash_attn=config.model.flash_attn
        ).to(device)
        
        print(f"Created {config.model.name} model with {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        raise RuntimeError(f"Failed to create model: {e}")

    # Load checkpoint
    try:
        state_dict = find_model(args.ckpt)
        model.load_state_dict(state_dict)
        print(f"✅ Loaded model from {args.ckpt}")
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    # Initialize diffusion
    try:
        diffusion = IaNDiffusion(timestep_respacing=str(args.num_sampling_steps),
                                    loss_type="l2")
        print(f"Initialized diffusion with {args.num_sampling_steps} sampling steps")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize diffusion: {e}")

    # Calculate number of batches needed
    num_batches = (args.total_samples + args.num_samples - 1) // args.num_samples
    print(f"Generating {args.total_samples} samples in {num_batches} batches...")
    
    # Generate samples in batches
    try:
        for batch_idx in tqdm(range(num_batches), desc="Generating samples"):
            with torch.no_grad():
                # Calculate samples for this batch
                current_batch_size = min(args.num_samples, args.total_samples - batch_idx * args.num_samples)
                
                # Generate noise
                z = torch.randn(current_batch_size, 
                              config.model.in_channels, 
                              config.data.image_size, 
                              config.data.image_size, 
                              config.data.image_size, 
                              device=device)
                
                # Sample from diffusion model
                try:
                    xs_samples, x0_samples = diffusion.p_sample_loop(
                        model.forward, z.shape, z,
                        clip_denoised=False,
                        progress=False,  # Disable internal progress bar
                        device=device,
                        new_sampling=True,
                        model_kwargs={"y": None} if config.model.num_classes else {}
                    )
                except Exception as e:
                    print(f"❌ Sampling failed for batch {batch_idx + 1}: {e}")
                    continue
                
                # Extract final samples
                print(f"XS range (before transform): {xs_samples[-1].min().item():.5f}, {xs_samples[-1].max().item():.5f}, {xs_samples[-1].std().item():.5f}")
                print(f"X0 range (before transform): {x0_samples.min().item():.5f}, {x0_samples.max().item():.5f},{x0_samples.std().item():.5f}")
                
                # Apply backward transformation to convert from [-1, 1] to [0, 1]
                xs_sample = data_transform_backward(xs_samples[-1])
                x0_samples = data_transform_backward(x0_samples)
                
                print(f"backward: {xs_sample.min().item():.5f}, {xs_sample.max().item():.5f}, {xs_sample.std().item():.5f}")
                print(f"backward: {x0_samples.min().item():.5f}, {x0_samples.max().item():.5f}, {x0_samples.std().item():.5f}")
                
                # Clamp to [0, 1] range
                xs_sample = torch.clamp(xs_sample, 0.0, 1.0)
                x0_samples = torch.clamp(x0_samples, 0.0, 1.0)
                
                print(f"After backward transform: XS range: {xs_sample.min().item():.5f}, {xs_sample.max().item():.5f}, {xs_sample.std().item():.5f}")
                print(f"After backward transform: X0 range: {x0_samples.min().item():.5f}, {x0_samples.max().item():.5f},{x0_samples.std().item():.5f}")
                
                if args.verbose:
                    print(f"Batch {batch_idx + 1}: XS range [{xs_sample.min().item():.3f}, {xs_sample.max().item():.3f}], "
                          f"X0 range [{x0_samples.min().item():.3f}, {x0_samples.max().item():.3f}]")
                
                # Calculate the starting index for this batch
                start_idx = batch_idx * args.num_samples
                
                # Save samples for this batch
                try:
                    from util import save_evaluation_samples
                    save_evaluation_samples(xs_sample, xs_path, config.data.image_size,
                                          epoch=start_idx, logger=None)
                    save_evaluation_samples(x0_samples, x0_path, config.data.image_size,
                                          epoch=start_idx, logger=None)
                    
                    if args.verbose:
                        print(f"✅ Saved batch {batch_idx + 1}/{num_batches} (samples {start_idx + 1}-{start_idx + current_batch_size})")
                        
                except Exception as e:
                    print(f"❌ Failed to save batch {batch_idx + 1}: {e}")
                    continue
        
        print(f"🎉 Completed generating {args.total_samples} samples. All samples saved in {args.output_dir}")
        
    except KeyboardInterrupt:
        print(f"⚠️  Sampling interrupted by user. Partial results saved in {args.output_dir}")
    except Exception as e:
        raise RuntimeError(f"Sampling failed: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a DiT model")
    parser.add_argument("--config", type=str, default="lidc_stage2_global.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Number of samples per batch")
    parser.add_argument("--total-samples", type=int, default=20,
                        help="Total number of samples to generate")
    parser.add_argument("--ckpt", type=str, default="results/001-DiT-B-12-12/checkpoints/013000.pt",
                        help="Path to the model checkpoint")
    parser.add_argument("--num-sampling-steps", type=int, default=1000,
                        help="Number of sampling steps")
    parser.add_argument("--output-dir", type=str, default="samples",
                        help="Directory to save samples")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed statistics during sampling")
    return parser.parse_args()


if __name__ == "__main__":
    main()

"""
Usage Examples:
    # Basic sampling:
    python sample.py --config lidc_stage2_global.yaml --ckpt path/to/checkpoint.pt
    
    # Custom sampling with verbose output:
    python sample.py --config lidc_stage2_global.yaml --ckpt path/to/checkpoint.pt \
                     --num-samples 8 --total-samples 100 \
                     --output-dir my_samples --verbose
    
    # High-quality sampling with more steps:
    python sample.py --config lidc_stage2_global.yaml --ckpt path/to/checkpoint.pt \
                     --num-sampling-steps 1000
"""
