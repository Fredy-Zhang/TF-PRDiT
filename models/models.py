"""
Diffusion Transformer (DiT) for 3D Volume Generation

A modular implementation of a Diffusion Transformer for generating 3D medical volumes.
The architecture supports two-stage training with both MLP-based coarse denoising
and Transformer-based fine refinement paths.

Key Features:
- Two-stage training (coarse MLP path + fine Transformer path)
- Normalized 3D positional embeddings
- AdaLN-Zero conditioning for stable training
- Flexible patch-based processing for 3D volumes
- Support for both depth=0 (MLP-only) and depth>0 (hybrid) models

Architecture Components:
1. PatchEmbed3D: 3D patch extraction and embedding
2. TimestepEmbedder: Sinusoidal timestep encoding with dual heads
3. CoarseDenoiser: MLP-based denoising path
4. FineRefiner: Transformer-based refinement path
5. DiTBlock: Transformer blocks with AdaLN conditioning

References:
- GLIDE: https://github.com/openai/glide-text2im
- MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
"""

# Standard library imports
import math
import logging
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import SwiGLU
from timm.models.vision_transformer import Mlp

from models.utils import _ntuple, modulate, get_normalized_3d_pos_enc
from models.classes import Attention, RMSNorm
from util import requires_grad

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

to_3tuple = _ntuple(3)


class ExtractPatches3D(nn.Module):
    def __init__(self, patch_size: Union[int, Tuple[int, int, int]], 
                 stride: Union[int, Tuple[int, int, int]], 
                 padding: int = 0):
        super().__init__()
        self.patch_size = to_3tuple(patch_size)
        self.stride = to_3tuple(stride)
        self.padding = padding
    
    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = volume.size()
        if self.padding > 0:
            volume = F.pad(volume, (self.padding,) * 6, mode='reflect')
        patches = (volume
                  .unfold(2, self.patch_size[0], self.stride[0])
                  .unfold(3, self.patch_size[1], self.stride[1])
                  .unfold(4, self.patch_size[2], self.stride[2]))
        patch_volume = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        num_patches = patches.numel() // (B * C * patch_volume)
        patches = (patches.contiguous()
                  .view(B, C, num_patches, patch_volume)
                  .permute(0, 2, 1, 3)
                  .reshape(B, num_patches, -1))
        return patches

    def compute_num_patches(self, input_size: Union[int, Tuple[int, int, int]]) -> Tuple[int, Tuple[int, int, int]]:
        input_size = to_3tuple(input_size)
        if self.padding > 0:
            input_size = tuple(s + 2 * self.padding for s in input_size)
        grid_size = tuple(
            ((s - p) // st) + 1 
            for s, p, st in zip(input_size, self.patch_size, self.stride)
        )
        return grid_size[0] * grid_size[1] * grid_size[2], grid_size

    def extra_repr(self) -> str:
        return f'patch_size={self.patch_size}, stride={self.stride}, padding={self.padding}'

class PatchEmbed3D(nn.Module):
    def __init__(self,
                 patch_size: int = 16,
                 in_chans: int = 1,
                 embed_dim: int = 768,
                 norm_layer: Optional[Callable] = nn.LayerNorm,
                 mlp_ratio: float = 4.0,
                 activation: Callable = nn.GELU(approximate="tanh"),
                 dropout: float = 0.0):
        super().__init__()
        input_dim = in_chans * (patch_size ** 3)
        hidden_dim = int(embed_dim * mlp_ratio)
        logger.debug(f"PatchEmbed3D: input_dim={input_dim}, hidden_dim={hidden_dim}, embed_dim={embed_dim}")
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.act = activation
        self.fc2 = nn.Linear(hidden_dim, embed_dim, bias=True)
        self.skip = nn.Linear(input_dim, embed_dim, bias=False)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act(h)
        h = self.fc2(h)
        s = self.skip(x)
        out = self.norm(h + s)
        return self.drop(out)

class TimestepEmbedder(nn.Module):
    def __init__(self, 
                 hidden_size: int, 
                 coarse_hidden_size: int, 
                 fine_hidden_size: int, 
                 frequency_embedding_size: int = 256,
                 is_depth_zero: bool = True):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU()
        )
        self.coarse_head = nn.Linear(hidden_size, coarse_hidden_size, bias=True)
        self.fine_head = (nn.Identity() if is_depth_zero 
                         else nn.Linear(hidden_size, fine_hidden_size, bias=True))

    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        timestep_emb = self.timestep_embedding(t, self.frequency_embedding_size)
        shared_features = self.mlp(timestep_emb)
        coarse_emb = self.coarse_head(shared_features)
        fine_emb = self.fine_head(shared_features)
        return coarse_emb, fine_emb

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

class DiTBlock(nn.Module):
    def __init__(self,
                 hidden_size: int, 
                 num_heads: int, 
                 mlp_ratio: float = 4.0, 
                 flash_attn: bool = False, 
                 norm_eps: float = 1e-5,
                 **block_kwargs):
        super().__init__()
        
        # Layer normalization (no learnable parameters - controlled by conditioning)
        # self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm1 = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=False)
        self.norm2 = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=False)
        
        # Core transformer components
        self.attn = Attention(
            hidden_size, 
            num_heads=num_heads, 
            qkv_bias=True, 
            use_flash_attention=flash_attn, 
            **block_kwargs
        )
        
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0
        )
        
        # AdaLN conditioning network (6 params: shift/scale/gate for attn and MLP)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
            RMSNorm(6 * hidden_size, eps=norm_eps)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size**3 * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)

class MlpDenoiser(nn.Module):
    def __init__(self, 
                 input_size: int,
                 hidden_size: int, 
                 patch_size: int, 
                 out_channels: int, 
                 mlp_ratio: float = 1.0,
                 swiglu_mlp: bool = False,
                 norm_eps: float = 1e-5):
        super().__init__()
        
        # Normalization layers (parameters controlled by conditioning)
        # self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm1 = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=False)
        self.norm2 = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=False)
        
        # First MLP block
        self.mlp1 = SwiGLU(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            norm_layer=nn.LayerNorm,
            drop=0,
        )
    
        # Second MLP block
        self.mlp2 = SwiGLU(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            norm_layer=nn.LayerNorm,
            drop=0,
        )
        self.linear_final = nn.Linear(hidden_size, patch_size**3 * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
            RMSNorm(6 * hidden_size, eps=norm_eps),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift1, scale1, shift2, scale2, shift3, scale3 = self.adaLN_modulation(c).chunk(6, dim=1)
        h = self.mlp1(modulate(self.norm1(x), shift1, scale1))
        h = self.mlp2(modulate(self.norm2(h), shift2, scale2))
        h = h + x
        h = modulate(h, shift3, scale3)
        return self.linear_final(h)

class CoarseDenoiser(nn.Module):
    def __init__(self,
                 in_channels: int,
                 extract_patch_size: int,
                 patch_size: int,
                 out_channels: int,
                 input_size: int,
                 stride: int = 4,
                 padding: int = 2,
                 mlp_ratio: float = 1.0,
                 swiglu_mlp: bool = True):
        super().__init__()
        self.patch_extractor = ExtractPatches3D(
            patch_size=extract_patch_size,
            stride=stride,
            padding=padding,
        )
        self.num_patches, self.grid_size = self.patch_extractor.compute_num_patches(input_size)
        input_dim = in_channels * extract_patch_size**3
        self.mlp_denoise = MlpDenoiser(
            input_size=input_size,
            hidden_size=input_dim,
            patch_size=patch_size,
            out_channels=out_channels,
            swiglu_mlp=swiglu_mlp,
            mlp_ratio=mlp_ratio
        )
    
    def forward(self, 
                x: torch.Tensor, 
                c: torch.Tensor, 
                return_patches: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        patches = self.patch_extractor(x)
        denoised = self.mlp_denoise(patches, c)
        return (patches, denoised) if return_patches else denoised

class FineRefiner(nn.Module):
    def __init__(self,
                 in_channels: int,
                 extract_patch_size: int,
                 hidden_size: int,
                 patch_size: int,
                 out_channels: int,
                 depth: int,
                 num_heads: int,
                 num_patches: int,
                 input_size: int,
                 stride: int = 4,
                 padding: int = 2,
                 mlp_ratio: float = 4.0,
                 flash_attn: bool = False):
        super().__init__()
        logger.debug(f"FineRefiner: depth={depth}, hidden_size={hidden_size}, num_heads={num_heads}")
        self.patch_embedder = PatchEmbed3D(
            patch_size=extract_patch_size,
            in_chans=in_channels, 
            embed_dim=hidden_size, 
            norm_layer=nn.LayerNorm,
            activation=nn.GELU(approximate="tanh")
        )
        grid_size = input_size // patch_size
        pos_embed = get_normalized_3d_pos_enc(grid_size=grid_size, embed_dim=hidden_size)
        self.register_buffer('pos_embed', pos_embed.unsqueeze(0), persistent=False)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, 
                     num_heads, 
                     mlp_ratio=mlp_ratio, 
                     flash_attn=flash_attn)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)
        self.input_size = input_size
        self.patch_size = patch_size
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.patch_embedder(x)
        h = h + self.pos_embed
        for block in self.blocks:
            h = block(h, c)
        return self.final_layer(h, c)

# =============================================================================
# Main DiT Model Architecture
# =============================================================================
class DiT(nn.Module):
    def __init__(self,
                 input_size: int = 32,
                 patch_size: int = 2,
                 stride: int = 4,
                 padding: int = 2,
                 in_channels: int = 1,
                 hidden_size: int = 1152,
                 depth: int = 28,
                 num_heads: int = 16,
                 mlp_ratio: float = 4.0,
                 class_dropout_prob: float = 0.1,
                 num_classes: int = 1,
                 learn_sigma: bool = False,
                 flash_attn: bool = False):
        super().__init__()
        
        # =====================================================================
        # Model Configuration
        # =====================================================================
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.input_size = input_size
        self.patch_size = stride
        self.depth = depth  
        self.hidden_size = hidden_size
        self._config_to_log = {
            'input_size': input_size, 'patch_size': patch_size, 'stride': stride,
            'in_channels': in_channels, 'hidden_size': hidden_size, 'depth': depth,
            'num_heads': num_heads, 'learn_sigma': learn_sigma
        }
        
        # =====================================================================
        # Timestep Embedding (Shared with Dual Heads)
        # =====================================================================
        self.t_embedder = TimestepEmbedder(
            hidden_size=hidden_size,
            coarse_hidden_size=int(in_channels * patch_size**3),
            fine_hidden_size=hidden_size,
            frequency_embedding_size=256,
            is_depth_zero=(depth == 0)
        )
        
        #---------------------------------------------------------------------
        # Denoising paths
        #---------------------------------------------------------------------
        # 1. Coarse path (MLP-based) - always present
        self.coarse = CoarseDenoiser(
            in_channels=in_channels,
            extract_patch_size=patch_size,
            patch_size=self.patch_size,
            out_channels=self.out_channels,
            input_size=input_size,
            stride=stride,
            padding=padding,
            mlp_ratio=1.0,
            swiglu_mlp=True
        )
        
        # Fine Path: Transformer-based refinement (only when depth > 0)
        self.fine = None
        if depth > 0:
            self.fine = FineRefiner(
                in_channels=in_channels,
                extract_patch_size=patch_size,
                hidden_size=hidden_size,
                patch_size=self.patch_size,
                out_channels=self.out_channels,
                depth=depth,
                num_heads=num_heads,
                num_patches=self.coarse.num_patches,
                input_size=input_size,
                stride=stride,
                padding=padding,
                mlp_ratio=mlp_ratio,
                flash_attn=flash_attn
            )
        
        # Initialize all weights
        self.initialize_weights()
        
        # In stage 2 (depth > 0), freeze the coarse path
        if depth > 0:
            self.freeze_coarse_path()
            logger.info(f"Stage 2 setup: Coarse path frozen, training {depth} transformer layers")

    def forward(
        self, 
        input: torch.Tensor, 
        t: torch.Tensor, 
        y: Optional[torch.Tensor] = None,
        return_intermediate: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        c_coarse, c_fine = self.t_embedder(t)
        with torch.no_grad() if self.depth > 0 else torch.enable_grad():
            if self.depth > 0 or return_intermediate:
                patches, coarse_out = self.coarse(input, c_coarse, return_patches=True)
            else:
                coarse_out = self.coarse(input, c_coarse)
        if self.depth > 0 and self.fine is not None:
            fine_out = self.fine(patches, c_fine)
            if return_intermediate:
                return self.unpatchify_3d(coarse_out), self.unpatchify_3d(fine_out)
            x = coarse_out + fine_out
        else:
            x = coarse_out
        x = self.unpatchify_3d(x)
        return x

    def unpatchify_3d(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.patch_size
        grid_size = self.input_size // p
        x = x.reshape(-1, grid_size, grid_size, grid_size, p, p, p, c)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return x.reshape(-1, c, grid_size * p, grid_size * p, grid_size * p)

    def freeze_coarse_path(self) -> None:
        requires_grad(self.coarse, False)
        requires_grad(self.t_embedder.coarse_head, False)
        requires_grad(self.t_embedder.mlp, False)
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Frozen {frozen_params:,} parameters, {trainable_params:,} trainable")

    def log_config(self, rank: int = 0) -> None:
        if rank == 0 and hasattr(self, '_config_to_log'):
            logger.info("DiT Model Configuration:")
            for key, value in self._config_to_log.items():
                logger.info(f"  {key}: {value}")

    def load_coarse_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(checkpoint['model'], strict=False)
        self.freeze_coarse_path()
        logger.info(f"Loaded stage 1 checkpoint from {checkpoint_path} and froze coarse path")

    def initialize_weights(self, gain: float = 1.0) -> None:
        logger.info("Initializing model weights...")
        def _init_linear_layers(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv3d):
                torch.nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_init_linear_layers)
        self._init_timestep_embedder()
        self._init_coarse_path()
        if self.depth > 0 and self.fine is not None:
            self._init_fine_path(gain)
        logger.info("Weight initialization complete")
    
    def _init_timestep_embedder(self) -> None:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        if self.t_embedder.mlp[0].bias is not None:
            nn.init.zeros_(self.t_embedder.mlp[0].bias)
        nn.init.normal_(self.t_embedder.coarse_head.weight, std=0.02)
        if self.t_embedder.coarse_head.bias is not None:
            nn.init.zeros_(self.t_embedder.coarse_head.bias)
        if not isinstance(self.t_embedder.fine_head, nn.Identity):
            nn.init.normal_(self.t_embedder.fine_head.weight, std=0.02)
            if self.t_embedder.fine_head.bias is not None:
                nn.init.zeros_(self.t_embedder.fine_head.bias)
    
    def _init_coarse_path(self) -> None:
        nn.init.constant_(self.coarse.mlp_denoise.adaLN_modulation[-2].weight, 0)
        nn.init.constant_(self.coarse.mlp_denoise.adaLN_modulation[-2].bias, 0)
        if hasattr(self.coarse.mlp_denoise, 'linear_final'):
            nn.init.constant_(self.coarse.mlp_denoise.linear_final.weight, 0)
            nn.init.constant_(self.coarse.mlp_denoise.linear_final.bias, 0)
        elif hasattr(self.coarse.mlp_denoise, 'linear_img'):
            nn.init.constant_(self.coarse.mlp_denoise.linear_img.weight, 0)
            nn.init.constant_(self.coarse.mlp_denoise.linear_img.bias, 0)
            if hasattr(self.coarse.mlp_denoise, 'linear_nos'):
                nn.init.constant_(self.coarse.mlp_denoise.linear_nos.weight, 0)
                nn.init.constant_(self.coarse.mlp_denoise.linear_nos.bias, 0)
    
    def _init_fine_path(self, gain: float) -> None:
        if hasattr(self.fine, 'patch_embedder'):
            nn.init.xavier_uniform_(self.fine.patch_embedder.fc1.weight, gain=gain)
            nn.init.xavier_uniform_(self.fine.patch_embedder.fc2.weight, gain=gain)
            nn.init.xavier_uniform_(self.fine.patch_embedder.skip.weight, gain=0.1)
            for layer in [self.fine.patch_embedder.fc1, self.fine.patch_embedder.fc2]:
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        for block in self.fine.blocks:
            nn.init.constant_(block.adaLN_modulation[-2].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-2].bias, 0)
        nn.init.constant_(self.fine.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.fine.final_layer.adaLN_modulation[-1].bias, 0)
        if hasattr(self.fine.final_layer, 'linear'):
            nn.init.constant_(self.fine.final_layer.linear.weight, 0)
            nn.init.constant_(self.fine.final_layer.linear.bias, 0)
        elif hasattr(self.fine.final_layer, 'linear_image'):
            nn.init.constant_(self.fine.final_layer.linear_image.weight, 0)
            nn.init.constant_(self.fine.final_layer.linear_image.bias, 0)
            if hasattr(self.fine.final_layer, 'linear_noise'):
                nn.init.constant_(self.fine.final_layer.linear_noise.weight, 0)
                nn.init.constant_(self.fine.final_layer.linear_noise.bias, 0)

