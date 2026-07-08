"""
Optimized sample processing utilities for CT volume handling.

This module provides optimized functions for:
- CT volume loading and preprocessing
- Subject creation and canonicalization
- HU to density transformation
- Sample extraction and intrinsic parameter handling
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union, List, Any, Tuple, Dict
import warnings

import numpy as np
import pandas as pd
import torch
import torch as th
import nibabel as nib
from torchio import LabelMap, ScalarImage, Subject
from torchio.transforms import Resample
from diffdrr.pose import RigidTransform

__all__ = ['saving_sample_to_extract_instrins', 'load_example_ct', 'read', 'canonicalize', 'transform_hu_to_density']

# ============================================================================
# MOST FREQUENTLY USED FUNCTIONS (Core functionality)
# ============================================================================

def saving_sample_to_extract_instrins(
    image: Union[th.Tensor, np.ndarray], 
    affine: Union[th.Tensor, np.ndarray], 
    save_path: Union[str, Path]
) -> Subject:
    """Save sample image and extract intrinsic parameters.
    
    This is the most frequently used function for creating sample subjects
    with proper intrinsic parameters for DRR generation.
    
    Args:
        image: Input image tensor or array [C, D, H, W] or [D, H, W]
        affine: Affine transformation matrix [4, 4]
        save_path: Path to save the NIfTI file
        
    Returns:
        Subject object with intrinsic parameters
    """
    # Convert to numpy arrays for nibabel compatibility
    if th.is_tensor(image):
        image = image.detach().cpu().numpy()
    if th.is_tensor(affine):
        affine = affine.detach().cpu().numpy()
    
    # Ensure image is 3D by taking first channel if 4D
    if len(image.shape) == 4:
        image = image[0]  # Take first channel
        
    # Create and save NIfTI image
    img = nib.Nifti1Image(image, affine)
    nib.save(img, save_path)
    
    # Load the saved image as a Subject object
    return load_example_ct(save_path)


def load_example_ct(
    volume: Union[str, Path, ScalarImage],
    labels: Optional[Union[int, List[int]]] = None,
    orientation: str = "AP",
    bone_attenuation_multiplier: float = 1.0,
    **kwargs
) -> Subject:
    """Load an example chest CT for demonstration purposes.
    
    Args:
        volume: Path to volume file or ScalarImage object
        labels: Optional labels for mask filtering
        orientation: Frame orientation ("AP", "PA", or None)
        bone_attenuation_multiplier: Multiplier for bone density
        **kwargs: Additional arguments passed to read()
        
    Returns:
        Subject object with loaded CT data
    """
    return read(
        volume,
        None,
        labels,
        orientation=orientation,
        bone_attenuation_multiplier=bone_attenuation_multiplier,
        **kwargs,
    )


def read(
    volume: Union[str, Path, ScalarImage],
    labelmap: Optional[Union[str, Path, LabelMap]] = None,
    labels: Optional[Union[int, List[int]]] = None,
    orientation: Optional[str] = "AP",
    bone_attenuation_multiplier: float = 1.0,
    fiducials: Optional[th.Tensor] = None,
    **kwargs
) -> Subject:
    """Read and process image volume with optional labelmap.
    
    Converts volume to RAS+ coordinate system and moves the volume 
    isocenter to the world origin.
    
    Args:
        volume: CT volume path or ScalarImage object
        labelmap: Optional labelmap for the volume
        labels: Labels from mask of structures to render
        orientation: Frame-of-reference change ("AP", "PA", or None)
        bone_attenuation_multiplier: Scalar multiplier on bone density
        fiducials: 3D fiducials in world coordinates
        **kwargs: Additional information for Subject
        
    Returns:
        Processed Subject object
    """
    # Read the volume
    if isinstance(volume, ScalarImage):
        pass
    else:
        volume = ScalarImage(volume)

    # Convert volume to density (HU to density transformation)
    density = volume.data  # transform_hu_to_density(volume.data, bone_attenuation_multiplier)
    density = ScalarImage(tensor=density, affine=volume.affine)

    # Read the mask if provided
    if labelmap is not None:
        if isinstance(labelmap, LabelMap):
            mask = labelmap
        else:
            mask = LabelMap(labelmap)
        _ = mask.data  # Load and cache the labelmap
    else:
        mask = None

    # Get reorientation matrix
    reorient = _get_reorientation_matrix(orientation)

    # Package the subject
    subject = Subject(
        volume=volume,
        mask=mask,
        reorient=reorient,
        density=density,
        fiducials=fiducials,
        **kwargs,
    )

    # Canonicalize the subject
    subject = canonicalize(subject)

    # Apply mask filtering if labels are specified
    if labels is not None:
        subject = _apply_mask_filtering(subject, labels)

    return subject


def canonicalize(subject: Subject) -> Subject:
    """Canonicalize subject by converting to RAS+ and centering at origin.
    
    Args:
        subject: Input subject to canonicalize
        
    Returns:
        Canonicalized subject
    """
    # Get the original affine matrix
    affine_original = th.from_numpy(subject.volume.affine)

    # Move the Subject's isocenter to the origin in world coordinates
    for image in subject.get_images(intensity_only=False):
        isocenter = image.get_center()
        Tinv = np.array([
            [1.0, 0.0, 0.0, -isocenter[0]],
            [0.0, 1.0, 0.0, -isocenter[1]],
            [0.0, 0.0, 1.0, -isocenter[2]],
            [0.0, 0.0, 0.0, 1.0],
        ])
        image.affine = Tinv.dot(image.affine)

    # Reorient fiducials if provided
    if subject.fiducials is not None:
        affine_new = th.tensor(image.affine)
        affine = affine_new @ affine_original.inverse()
        affine = affine.to(subject.fiducials)
        affine = RigidTransform(affine)
        subject.fiducials = affine(subject.fiducials)
    
    return subject


# ============================================================================
# DENSITY TRANSFORMATION FUNCTIONS
# ============================================================================

def transform_hu_to_density(
    volume: th.Tensor, 
    bone_attenuation_multiplier: float = 1.0
) -> th.Tensor:
    """Transform Hounsfield Units to density values.
    
    Args:
        volume: Input volume in HU
        bone_attenuation_multiplier: Multiplier for bone density
        
    Returns:
        Density volume normalized to [0, 1]
    """
    # Convert to float32 for proper computation
    volume = volume.to(th.float32)
    
    # Define tissue regions based on HU values
    air_mask = volume <= -800
    soft_tissue_mask = (volume > -800) & (volume <= 350)
    bone_mask = volume > 350

    # Initialize density tensor
    density = th.empty_like(volume)
    
    # Apply density transformations
    density[air_mask] = volume[soft_tissue_mask].min() if soft_tissue_mask.any() else volume.min()
    density[soft_tissue_mask] = volume[soft_tissue_mask]
    density[bone_mask] = volume[bone_mask] * bone_attenuation_multiplier
    
    # Normalize to [0, 1]
    density -= density.min()
    density /= density.max()
    
    return density


# ============================================================================
# HELPER FUNCTIONS (Internal use)
# ============================================================================

def _get_reorientation_matrix(orientation: Optional[str]) -> th.Tensor:
    """Get reorientation matrix for frame-of-reference change.
    
    Args:
        orientation: Orientation string ("AP", "PA", or None)
        
    Returns:
        Reorientation matrix [4, 4]
    """
    reorientation_matrices = {
        "AP": th.tensor([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
        "PA": th.tensor([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
        None: th.eye(4)
    }
    
    if orientation not in reorientation_matrices:
        raise ValueError(f"Unrecognized orientation {orientation}")
    
    return reorientation_matrices[orientation]


def _apply_mask_filtering(subject: Subject, labels: Union[int, List[int]]) -> Subject:
    """Apply mask filtering based on specified labels.
    
    Args:
        subject: Input subject
        labels: Labels to filter
        
    Returns:
        Subject with applied mask filtering
    """
    if isinstance(labels, int):
        labels = [labels]
    
    # Create combined mask for all specified labels
    mask = th.any(
        th.stack([subject.mask.data.squeeze() == idx for idx in labels]), 
        dim=0
    )
    
    # Apply mask to density
    subject.density.data = subject.density.data * mask
    
    return subject


# ============================================================================
# UTILITY FUNCTIONS (Less frequently used)
# ============================================================================

def create_sample_subject(
    image_data: th.Tensor,
    affine: th.Tensor,
    orientation: str = "AP",
    bone_attenuation_multiplier: float = 1.0
) -> Subject:
    """Create a sample subject from image data and affine matrix.
    
    Args:
        image_data: Image tensor [C, D, H, W] or [D, H, W]
        affine: Affine transformation matrix [4, 4]
        orientation: Frame orientation
        bone_attenuation_multiplier: Bone density multiplier
        
    Returns:
        Sample subject
    """
    # Ensure proper tensor format
    if len(image_data.shape) == 3:
        image_data = image_data.unsqueeze(0)  # Add channel dimension
    
    # Create temporary file for processing
    temp_path = "temp_sample.nii.gz"
    try:
        return saving_sample_to_extract_instrins(
            image_data, affine, temp_path
        )
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def validate_subject(subject: Subject) -> bool:
    """Validate that a subject has all required components.
    
    Args:
        subject: Subject to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check required components
        assert hasattr(subject, 'volume'), "Subject missing volume"
        assert hasattr(subject, 'density'), "Subject missing density"
        assert subject.volume.data is not None, "Volume data is None"
        assert subject.density.data is not None, "Density data is None"
        
        # Check data consistency
        assert subject.volume.data.shape == subject.density.data.shape, \
            "Volume and density shape mismatch"
        
        return True
    except (AttributeError, AssertionError) as e:
        warnings.warn(f"Subject validation failed: {e}")
        return False


def get_subject_info(subject: Subject) -> Dict[str, Any]:
    """Get information about a subject.
    
    Args:
        subject: Subject to analyze
        
    Returns:
        Dictionary with subject information
    """
    info = {
        'volume_shape': subject.volume.data.shape,
        'density_shape': subject.density.data.shape,
        'volume_range': (subject.volume.data.min().item(), subject.volume.data.max().item()),
        'density_range': (subject.density.data.min().item(), subject.density.data.max().item()),
        'has_mask': subject.mask is not None,
        'has_fiducials': subject.fiducials is not None,
    }
    
    if subject.mask is not None:
        info['mask_shape'] = subject.mask.data.shape
        info['unique_labels'] = th.unique(subject.mask.data).tolist()
    
    return info


if __name__ == "__main__":
    pass