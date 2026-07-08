"""
This module contains classes adapted from the timm (PyTorch Image Models) package.
Original source: https://github.com/huggingface/pytorch-image-models

These classes provide various patch embedding implementations used in Vision Transformers
and related architectures.
"""

import torch
import torch.nn as nn
from timm.layers import DropPath, to_2tuple, trunc_normal_, _assert

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    
    Originally from timm (PyTorch Image Models) package.
    This class implements overlapping patch embedding used in models like PVT (Pyramid Vision Transformer).
    
    Args:
        patch_size (int): Patch size for embedding. Default: 7
        stride (int): Stride size for the convolution. Default: 4
        in_chans (int): Number of input channels. Default: 3
        embed_dim (int): Embedding dimension. Default: 768
    """
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        assert max(patch_size) > stride, "Set larger patch_size than stride"
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim, patch_size,
            stride=stride, padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    
    Originally from timm (PyTorch Image Models) package.
    This class implements the patch embedding layer used in Vision Transformer (ViT) and related models.
    
    Args:
        img_size (int): Input image size. Default: 224
        patch_size (int): Patch size for embedding. Default: 16
        in_chans (int): Number of input channels. Default: 3
        embed_dim (int): Embedding dimension. Default: 768
        multi_conv (bool): Whether to use multiple convolutions for patch embedding. Default: False
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, multi_conv=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if multi_conv:
            if patch_size[0] == 12:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=3, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1),
                )
            elif patch_size[0] == 16:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
                )
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        _assert(H == self.img_size[0],
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}).")
        _assert(W == self.img_size[1],
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}).")
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution + batch norm
    
    Helper function from timm package for creating 3x3 convolution with batch normalization.
    
    Args:
        in_planes (int): Number of input channels
        out_planes (int): Number of output channels
        stride (int): Stride size for convolution. Default: 1
    
    Returns:
        nn.Sequential: A sequential container of Conv2d and BatchNorm2d layers
    """
    return torch.nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_planes)
    )
    
class ConvPatchEmbed(nn.Module):
    """Image to Patch Embedding using multiple convolutional layers
    
    Originally from timm (PyTorch Image Models) package.
    This class implements a convolutional patch embedding that uses multiple conv layers
    instead of a single large convolution.
    
    Args:
        img_size (int): Input image size. Default: 224
        patch_size (int): Patch size for embedding (must be 8 or 16). Default: 16
        in_chans (int): Number of input channels. Default: 3
        embed_dim (int): Embedding dimension. Default: 768
        act_layer (nn.Module): Activation layer. Default: nn.GELU
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, act_layer=nn.GELU):
        super().__init__()
        img_size = to_2tuple(img_size)
        num_patches = (img_size[1] // patch_size) * (img_size[0] // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        if patch_size == 16:
            self.proj = torch.nn.Sequential(
                conv3x3(in_chans, embed_dim // 8, 2),
                act_layer(),
                conv3x3(embed_dim // 8, embed_dim // 4, 2),
                act_layer(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                act_layer(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        elif patch_size == 8:
            self.proj = torch.nn.Sequential(
                conv3x3(in_chans, embed_dim // 4, 2),
                act_layer(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                act_layer(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        else:
            raise('For convolutional projection, patch size has to be in [8, 16]')

    def forward(self, x):
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x, (Hp, Wp)