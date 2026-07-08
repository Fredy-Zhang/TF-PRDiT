import numpy as np
import torch
import collections.abc
from itertools import repeat
from typing import Optional
import math

import logging
from einops import rearrange
logger = logging.getLogger(__name__)


# ============================================================================
# Basic Utility Functions
# ============================================================================

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_3tuple = _ntuple(3)


# ============================================================================
# Modulation Functions
# ============================================================================

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ============================================================================
# Positional Encoding Functions
# ============================================================================

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0
    
    emb_x = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])
    emb_y = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])
    emb_z = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])
    
    emb = np.concatenate([emb_x, emb_y, emb_z], axis=1)
    return emb


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_x = np.arange(grid_size, dtype=np.float32)
    grid_y = np.arange(grid_size, dtype=np.float32)
    grid_z = np.arange(grid_size, dtype=np.float32)

    grid = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([3, 1, grid_size, grid_size, grid_size])
    
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

"""version 1
def get_normalized_3d_pos_enc(grid_size: int, 
                              embed_dim: int, 
                              num_frequencies: Optional[int] = None) -> torch.Tensor:
    if num_frequencies is None:
        if embed_dim % 6 != 0:
            raise ValueError("embed_dim must be divisible by 6 or num_frequencies must be provided")
        num_frequencies = embed_dim // 6
    
    logger.debug(f"Generating 3D pos encoding: grid_size={grid_size}, embed_dim={embed_dim}, num_freq={num_frequencies}")
    
    # Generate normalized coordinates [0,1] with center offset
    coords = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
    
    # Create 3D grid
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing='ij')
    pos = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)  # [N, 3]
    
    # Compute frequency bands (avoiding aliasing)
    safety_factor = 0.95
    f_min, f_max = 1.0, safety_factor * (grid_size / 2.0)
    t = torch.linspace(0.0, 1.0, num_frequencies, dtype=torch.float32)
    freqs = f_min * (f_max / f_min) ** t * 2.0 * math.pi
    
    # Generate sin/cos encodings for each spatial dimension
    encodings = []
    for dim in range(3):
        for fn in [torch.sin, torch.cos]:
            encodings.append(fn(pos[:, dim:dim+1] * freqs))
    
    pos_enc = torch.cat(encodings, dim=-1)  # [N, 6*num_frequencies]
    
    if pos_enc.shape[-1] != embed_dim:
        raise ValueError(f"Encoding dimension mismatch: got {pos_enc.shape[-1]}, expected {embed_dim}")
    
    return pos_enc
"""

def get_normalized_3d_pos_enc(
    grid_size: int,
    embed_dim: int,
    num_frequencies: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    scale_freqs: float = 1.0  # Optional: multiply freqs by this for custom aliasing control
) -> torch.Tensor:
    if num_frequencies is None:
        if embed_dim % 6 != 0:
            raise ValueError("embed_dim must be divisible by 6 or num_frequencies must be provided")
        num_frequencies = embed_dim // 6

    depth, height, width = grid_size, grid_size, grid_size
    # Optional logger
    try:
        logger = logging.getLogger(__name__)
        logger.debug(f"Generating 3D pos encoding: D={depth}, H={height}, W={width}, embed_dim={embed_dim}, num_freq={num_frequencies}")
    except:
        pass  # No-op if logging not configured

    # Generate normalized coordinates [0,1] with center offset
    z_coords = (torch.arange(depth, dtype=dtype) + 0.5) / depth
    y_coords = (torch.arange(height, dtype=dtype) + 0.5) / height
    x_coords = (torch.arange(width, dtype=dtype) + 0.5) / width

    # Create 3D grid (indexing='ij' for row-major flattening)
    zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    pos = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)  # [N, 3]; x,y,z order

    # Compute frequency bands (geometric progression, anti-aliased)
    safety_factor = 0.95
    # Use effective grid size for max freq (min of dims to avoid over-high freqs in small dims)
    effective_size = min(depth, height, width)
    f_min, f_max = 1.0, safety_factor * (effective_size / 2.0)
    t = torch.linspace(0.0, 1.0, num_frequencies, dtype=dtype)
    freqs = f_min * (f_max / f_min) ** t * 2.0 * math.pi * scale_freqs

    # Vectorized sin/cos: broadcast pos * freqs, then apply sin/cos
    # Shape: [N, 3, num_freq] -> apply sin/cos -> [N, 3, 2*num_freq] -> [N, 6*num_freq]
    pos_expanded = pos.unsqueeze(-1) * freqs.unsqueeze(0)  # [N, 3, num_freq]
    sin_enc = torch.sin(pos_expanded)  # [N, 3, num_freq]
    cos_enc = torch.cos(pos_expanded)  # [N, 3, num_freq]
    
    # Concat per dim: sin_x + cos_x + sin_y + cos_y + sin_z + cos_z
    encodings = []
    for dim in range(3):
        encodings.append(sin_enc[:, dim])  # [N, num_freq]
        encodings.append(cos_enc[:, dim])  # [N, num_freq]
    pos_enc = torch.cat(encodings, dim=-1)  # [N, 6*num_freq]

    if pos_enc.shape[-1] != embed_dim:
        raise ValueError(f"Encoding dimension mismatch: got {pos_enc.shape[-1]}, expected {embed_dim}")

    return pos_enc

# ============================================================================
# Patch/Unpatch Functions
# ============================================================================

def unpatchify_3d(x: torch.Tensor, 
                  out_channels: int, 
                  patch_size: int, 
                  input_size: int) -> torch.Tensor:
    c = out_channels
    p = patch_size
    grid_size = input_size // p
    
    # Reshape to 3D grid
    x = x.reshape(-1, grid_size, grid_size, grid_size, p, p, p, c)
    
    # Permute to get channels first and combine spatial dimensions
    # [B, D', H', W', p, p, p, C] -> [B, C, D', p, H', p, W', p]
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
    
    # Merge grid and patch dimensions: [B, C, D'*p, H'*p, W'*p]
    return x.reshape(-1, c, grid_size * p, grid_size * p, grid_size * p)


# ============================================================================
# Regularization Functions
# ============================================================================

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')