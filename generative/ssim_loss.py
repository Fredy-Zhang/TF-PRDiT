"""
SSIM Loss wrapper using MONAI's production-ready SSIMLoss implementation.
MONAI's SSIMLoss is optimized for medical imaging and supports 3D natively.

Reference: https://github.com/Project-MONAI/MONAI/blob/main/monai/losses/ssim_loss.py
"""

from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import SSIMLoss as MONAISSIMLoss
from monai.metrics.regression import KernelType


class SSIMLoss(nn.Module):
    """
    SSIM Loss wrapper using MONAI's implementation.
    Optimized for 3D medical images and sharpness enhancement.
    
    This is a wrapper around MONAI's SSIMLoss that:
    1. Returns per-sample loss [B] instead of scalar
    2. Handles input normalization automatically
    3. Provides a consistent interface with our training code
    
    Args:
        window_size: Size of Gaussian window (default: 11, optimal for 128×128×128)
        spatial_dims: Number of spatial dimensions (3 for 3D)
        data_range: Value range of input images (1.0 for normalized [0,1])
        kernel_sigma: Standard deviation for Gaussian kernel (default: 1.5)
    """
    
    def __init__(
        self,
        window_size: int = 11,
        spatial_dims: int = 3,
        data_range: float = 1.0,
        kernel_sigma: float = 1.5,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.data_range = data_range
        
        # Use MONAI's SSIMLoss - production-ready and optimized
        self.monai_ssim = MONAISSIMLoss(
            spatial_dims=spatial_dims,
            data_range=data_range,
            kernel_type=KernelType.GAUSSIAN,
            win_size=window_size,
            kernel_sigma=kernel_sigma,
            k1=0.01,  # Standard SSIM constants
            k2=0.03,
            reduction="none",  # Return per-sample loss
        )
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss (1 - SSIM) for sharpness and detail enhancement.
        
        Args:
            img1: First image tensor [B, C, D, H, W] (values in [0, 1])
            img2: Second image tensor [B, C, D, H, W] (values in [0, 1])
        
        Returns:
            SSIM loss per sample [B]
        """
        if img1.shape != img2.shape:
            raise ValueError(f"Input shapes must match: {img1.shape} vs {img2.shape}")
        
        # Ensure values are in [0, 1] range
        img1 = torch.clamp(img1, 0.0, 1.0)
        img2 = torch.clamp(img2, 0.0, 1.0)
        
        # MONAI SSIMLoss returns [B, 1] with reduction="none"
        # We need [B], so squeeze the last dimension
        loss = self.monai_ssim(img1, img2)
        
        # Handle shape: MONAI returns [B, 1] or [B] depending on version
        if loss.dim() > 1:
            loss = loss.squeeze(-1)
        
        return loss


class MultiScaleSSIMLoss(nn.Module):
    """
    Multi-Scale SSIM Loss using MONAI's SSIMLoss.
    Better for capturing both fine details and larger structures.
    Optimized for sharpness enhancement.
    
    Args:
        window_size: Base window size (default: 11)
        spatial_dims: Number of spatial dimensions (3 for 3D)
        scales: Number of scales (default: 3)
        weights: Weights for each scale (default: [0.5, 0.3, 0.2])
        data_range: Value range of input images (1.0 for normalized [0,1])
    """
    
    def __init__(
        self,
        window_size: int = 11,
        spatial_dims: int = 3,
        scales: int = 3,
        weights: Optional[List[float]] = None,
        data_range: float = 1.0,
    ):
        super().__init__()
        self.scales = scales
        self.weights = weights if weights is not None else [0.5, 0.3, 0.2][:scales]
        
        if len(self.weights) != scales:
            raise ValueError(f"Number of weights ({len(self.weights)}) must match scales ({scales})")
        
        # Normalize weights
        weight_sum = sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]
        
        # Create SSIM losses for each scale
        self.ssim_losses = nn.ModuleList([
            SSIMLoss(
                window_size=window_size,
                spatial_dims=spatial_dims,
                data_range=data_range
            )
            for _ in range(scales)
        ])
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale SSIM loss.
        
        Args:
            img1: First image tensor [B, C, D, H, W]
            img2: Second image tensor [B, C, D, H, W]
        
        Returns:
            Multi-scale SSIM loss per sample [B]
        """
        loss = torch.zeros(img1.shape[0], device=img1.device, dtype=img1.dtype)
        
        for scale, (ssim_loss, weight) in enumerate(zip(self.ssim_losses, self.weights)):
            if scale > 0:
                # Downsample for multi-scale
                img1_scaled = F.avg_pool3d(img1, kernel_size=2, stride=2)
                img2_scaled = F.avg_pool3d(img2, kernel_size=2, stride=2)
            else:
                img1_scaled = img1
                img2_scaled = img2
            
            scale_loss = ssim_loss(img1_scaled, img2_scaled)
            loss += weight * scale_loss
        
        return loss

