import os
import time
import math
import random
import warnings
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import cv2
from torchio import ScalarImage, Subject
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from diffdrr.data import load_example_ct
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform
from diffdrr.visualization import plot_drr
from conds.sample import saving_sample_to_extract_instrins
from models.utils import _ntuple
from datasets import get_voxel_dataset

from torchio import ScalarImage, Subject as TorchioSubject
import SimpleITK as sitk

logger = logging.getLogger(__name__)

#------------------------------------------------#
DEFAULT_SHAPE = (256, 256, 256)
DEFAULT_SPACING = (1.0, 1.0, 1.0)
MAX_HU = 2500
SDD = 820.0
HEIGHT = 128
SAD = 700.0
DET_HEIGHT_PX = 128
PIXEL_SIZE_MM = 1.0
#------------------------------------------------#

class DiffDRRSubject(TorchioSubject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reorient = torch.eye(4)
        self.mask = None

def get_xrays_from_ct(
    data: torch.Tensor,
    idx: int,
    rotations: int = 2,
    device: str = "cpu",
    convention: str = "XYZ",
    save_dir: Optional[str] = "Intermediate_Results",
    height: int = 128,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    apply_density_transform: bool = False,
    is_normalized: bool = True,
    hu_min: float = 0.0,
    hu_max: float = 4096.0,
) -> torch.Tensor:
    logger.info(f"=" * 60)
    logger.info(f"Generating X-rays from CT data (index: {idx})")
    logger.info(f"  Input data shape: {data.shape}, dtype: {data.dtype}")
    logger.info(f"  Data value range: min={data.min().item():.4f}, max={data.max().item():.4f}, mean={data.mean().item():.4f}")
    
    # data = correct_data_shape(data)    
    logger.info(f"  Constructing subject with:")
    logger.info(f"    spacing={spacing}, orientation='AP'")
    logger.info(f"    apply_density_transform={apply_density_transform}")
    logger.info(f"    is_normalized={is_normalized}, hu_range=[{hu_min}, {hu_max}]")
    
    subject = construct_subject(
        volume_data=data,
        spacing=spacing,
        orientation="AP",
        apply_density_transform=apply_density_transform,
        is_normalized=is_normalized,
        hu_min=hu_min,
        hu_max=hu_max
    )
    logger.info(f"  Subject constructed successfully")

    if save_dir is not None and not data.requires_grad:
        logger.info(f"  Saving middle slices to: {save_dir}")
        vol_for_save = data
        if vol_for_save.dim() == 4:
            vol_for_save = vol_for_save.unsqueeze(0)
        if vol_for_save.dim() != 5:
            raise ValueError(f"Unexpected data shape for saving: {vol_for_save.shape}")
        save_middle_slices(volume=vol_for_save, step=f"ct_idx_{idx}", save_dir=save_dir)
        logger.info(f"  Middle slices saved")

    logger.info(f"  Calling ct_to_xrays...")
    xrays = ct_to_xrays(subject, idx, 
                        rotations, 
                        device, 
                        convention=convention, 
                        height=height)
    logger.info(f"  X-ray generation from CT completed successfully")
    logger.info(f"=" * 60)
    
    return xrays.detach().cpu() if xrays.device.type != 'cpu' else xrays

def ct_to_xrays(
    subject: Subject,
    idx: int,
    rotations: int = 2,
    device: str = "cpu",
    convention: str = "XYZ",
    height: int = 128,
) -> torch.Tensor:
    logger.info(f"Starting X-ray generation for index {idx}")
    logger.info(f"  Device: {device}, Rotations: {rotations}, Convention: {convention}, Height: {height}")
    
    if hasattr(subject, 'volume') and hasattr(subject.volume, 'shape'):
        logger.info(f"  Subject volume shape: {subject.volume.shape}")
    if hasattr(subject, 'volume') and hasattr(subject.volume, 'spacing'):
        logger.info(f"  Subject volume spacing: {subject.volume.spacing}")
    
    logger.info(f"  Creating DRR renderer with SDD={SDD}, height={height}, width={height}, delx={PIXEL_SIZE_MM}, dely={PIXEL_SIZE_MM}, renderer='siddon'")
    t_start = time.time()
    drr = DRR(subject, 
              sdd=SDD,
              height=height,
              width=height,
              delx=PIXEL_SIZE_MM,
              dely=PIXEL_SIZE_MM,
              renderer="siddon").to(device)
    logger.info(f"  DRR renderer created and moved to {device} in {time.time() - t_start:.2f}s")
    
    base = math.pi 
    if rotations == 2:
        angles_deg = [0.0, 90.0]
        angles_rad = [float(math.radians(base - angle)) for angle in angles_deg]
        rotations_list = [[0.0, angle_rad, 0.0] for angle_rad in angles_rad]
        logger.info(f"  Using 2-view setup: angles_deg={angles_deg}")
    elif rotations == 1:
        angles_deg = [0.0]
        angles_rad = [float(math.radians(base - angle)) for angle in angles_deg]
        rotations_list = [[0.0, angle_rad, 0.0] for angle_rad in angles_rad]
        logger.info(f"  Using single-view setup: angle_deg={angles_deg}")
    elif rotations > 2:
        angles_rad = [2.0 * math.pi * i / rotations for i in range(rotations)]
        rotations_list = [[0.0, angle, 0.0] for angle in angles_rad]
        logger.info(f"  Using {rotations} views: angles distributed evenly around circle")
    else:
        raise ValueError(f"Invalid number of rotations: {rotations}")
    
    rotations_tensor = torch.tensor(rotations_list, device=device, dtype=torch.float32)
    
    translations_tensor = torch.tensor([[0.0, 0.0, SAD]], device=device, dtype=torch.float32)
    
    logger.info(f"  Rotation tensor shape: {rotations_tensor.shape}, Translation: {translations_tensor[0].tolist()}")
    logger.info(f"  Rendering X-rays with parameterization='euler_angles', convention='{convention}'")
    
    t_start = time.time()
    img = drr(rotations_tensor, 
              translations_tensor,
              parameterization="euler_angles", 
              convention=convention)
    logger.info(f"  DRR generation completed in {time.time() - t_start:.2f}s")
    
    logger.info(f"  X-ray generation completed. Output shape: {img.shape}, dtype: {img.dtype}")
    logger.info(f"  X-ray value range: min={img.min().item():.4f}, max={img.max().item():.4f}, mean={img.mean().item():.4f}")
    
    if img.max() <= 0 or torch.allclose(img, torch.zeros_like(img)):
        logger.warning(f"  WARNING: DRR output appears to be empty or all zeros!")
        logger.warning(f"  This might indicate incorrect camera positioning or volume orientation")
        if hasattr(subject, 'volume'):
            logger.warning(f"  Subject volume shape: {subject.volume.shape}")
            logger.warning(f"  Subject volume spacing: {getattr(subject.volume, 'spacing', 'N/A')}")
            logger.warning(f"  Subject volume value range: min={subject.volume.data.min().item():.4f}, max={subject.volume.data.max().item():.4f}")
        if hasattr(subject, 'density'):
            logger.warning(f"  Subject density shape: {subject.density.shape}")
            logger.warning(f"  Subject density value range: min={subject.density.data.min().item():.4f}, max={subject.density.data.max().item():.4f}")
        logger.warning(f"  Rotation tensor: {rotations_tensor.tolist()}")
        logger.warning(f"  Translation: {translations_tensor[0].tolist()}")
    
    return img

def construct_subject(
    volume_data: torch.Tensor,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Subject:
    logger.info(f"  Constructing subject from volume data...")
    
    if len(volume_data.shape) == 5:
        volume_data = volume_data.squeeze(0)
    if len(volume_data.shape) == 3:
        volume_data = volume_data.unsqueeze(0)
    
    if len(volume_data.shape) != 4:
        raise ValueError(f"Expected 4D tensor [C, D, H, W], got shape {volume_data.shape}")
    
    C, D, H, W = volume_data.shape
    logger.info(f"  Input volume shape: {volume_data.shape} (C={C}, D={D}, H={H}, W={W})")
    
    volume_np = volume_data[0].cpu().numpy()
    logger.info(f"  Volume as numpy: shape={volume_np.shape}, range=[{volume_np.min():.2f}, {volume_np.max():.2f}]")
    
    if is_normalized:
        logger.info(f"  Converting normalized values to HU range [{hu_min}, {hu_max}]")
        volume_np = convert_normalized_to_hu(torch.from_numpy(volume_np), hu_min, hu_max).numpy()
        logger.info(f"  After HU conversion: range=[{volume_np.min():.1f}, {volume_np.max():.1f}]")
    
    density_scan = (np.clip(volume_np, 0, MAX_HU) / MAX_HU).astype(np.float32)
    logger.info(f"  Density range: [{density_scan.min():.6f}, {density_scan.max():.6f}]")
    
    density_scan = density_scan.transpose(2, 0, 1)
    volume_shape_xyz = density_scan.shape
    logger.info(f"  Transposed volume shape: {volume_shape_xyz} (X, Z, Y)")
    
    spacing_xyz = np.array([spacing[2], spacing[0], spacing[1]], dtype=np.float32)
    logger.info(f"  Adjusted spacing: {spacing_xyz} (X, Z, Y)")
    
    affine = get_centered_affine(spacing_xyz, volume_shape_xyz)
    logger.info(f"  Affine origin: {affine[:3, 3]}")
    
    vol_t = torch.from_numpy(density_scan)[None]
    logger.info(f"  Created tensor with shape: {vol_t.shape}")
    
    density_image = ScalarImage(tensor=vol_t, affine=affine)
    volume_image = ScalarImage(tensor=vol_t, affine=affine)
    
    subject = Subject(
        volume=volume_image,
        density=density_image,
        mask=mask,
        fiducials=fiducials,
    )
    
    subject.reorient = torch.eye(4, dtype=torch.float32)
    if not hasattr(subject, 'mask') or subject.mask is None:
        subject.mask = None
    
    logger.info(f"  TorchIO Subject created successfully")
    
    return subject

def construct_subject_from_ct(ct_input: Union[str, np.ndarray, torch.Tensor]):
    is_torch = isinstance(ct_input, torch.Tensor)
    
    if isinstance(ct_input, str):
        img_obj = sitk.ReadImage(ct_input)
        volume = sitk.GetArrayFromImage(img_obj)
        volume = np.array(volume, dtype=np.float32)
    elif isinstance(ct_input, torch.Tensor):
        volume = ct_input
    elif isinstance(ct_input, np.ndarray):
        volume = ct_input.copy().astype(np.float32)
    else:
        raise TypeError(f"Unsupported input type: {type(ct_input)}. Expected str, np.ndarray, or torch.Tensor")
    
    if is_torch:
        if volume.ndim == 3:
            volume = volume.unsqueeze(0)
        if volume.ndim == 5:
            volume = volume.squeeze(0)
        
        if volume.ndim != 4:
            raise ValueError(f"Expected 3D volume, got {volume.ndim}D tensor with shape {volume.shape}")
        
        density = volume.permute(0, 3, 1, 2)
        
        current_shape = tuple(density.shape[-3:])
        
        if current_shape == DEFAULT_SHAPE:
            new_spacing = DEFAULT_SPACING
        else:
            default_physical_size = torch.tensor(DEFAULT_SHAPE, dtype=torch.float32) * torch.tensor(DEFAULT_SPACING, dtype=torch.float32)
            scale_factor = default_physical_size / torch.tensor(current_shape, dtype=torch.float32)
            new_spacing = tuple(scale_factor.cpu().numpy())
        
        density_tensor = density
        affine = get_centered_affine(new_spacing, current_shape)
        
    else:
        if volume.ndim == 3:
            volume = volume.unsqueeze(0)
        if volume.ndim == 5:
            volume = volume.squeeze(0)
        
        if volume.ndim != 4:
            raise ValueError(f"Expected 3D volume, got {volume.ndim}D array with shape {volume.shape}")
        
        density = volume
        density = density.transpose(0, 3, 1, 2)
        
        current_shape = tuple(density.shape[-3:])
        
        if current_shape == DEFAULT_SHAPE:
            new_spacing = DEFAULT_SPACING
        else:
            default_physical_size = np.array(DEFAULT_SHAPE, dtype=np.float32) * np.array(DEFAULT_SPACING, dtype=np.float32)
            scale_factor = default_physical_size / np.array(current_shape, dtype=np.float32)
            new_spacing = tuple(scale_factor)
        
        density_tensor = torch.tensor(density).float()
        affine = get_centered_affine(new_spacing, current_shape)
    
    image_obj = ScalarImage(tensor=density_tensor, affine=affine)
    return DiffDRRSubject(volume=image_obj, density=image_obj)

def canonicalize(subject: Subject) -> Subject:
    affine_original = subject.volume.affine

    for image in subject.get_images(intensity_only=False):
        isocenter = image.get_center()
        Tinv = np.array([
            [1.0, 0.0, 0.0, -isocenter[0]],
            [0.0, 1.0, 0.0, -isocenter[1]],
            [0.0, 0.0, 1.0, -isocenter[2]],
            [0.0, 0.0, 0.0, 1.0],
        ])
        image.affine = Tinv.dot(image.affine)

    if subject.fiducials is not None:
        affine_new = torch.tensor(image.affine)
        affine = affine_new @ affine_original.inverse()
        affine = affine.to(subject.fiducials)
        affine = RigidTransform(affine)
        subject.fiducials = affine(subject.fiducials)
    
    return subject

def correct_data_shape(data: torch.Tensor) -> torch.Tensor:
    if data.dim() == 5:
        data = data.squeeze(0)
    assert data.dim() == 4 and data.shape[0] == 1, f"expected [1,D,H,W], got {tuple(data.shape)}"
    data = torch.flip(data, dims=[-1])
    return data

def convert_normalized_to_hu(
    volume: torch.Tensor,
    hu_min: float = 0.0,
    hu_max: float = 4096.0
) -> torch.Tensor:
    volume_01 = (torch.clamp(volume, min=-1.0, max=1.0) + 1.0) / 2.0
    volume_hu = volume_01 * (hu_max - hu_min) + hu_min
    return volume_hu

def transform_hu_to_density(
    volume: Union[np.ndarray, torch.Tensor], 
    bone_attenuation_multiplier: float = 1.0
) -> Union[np.ndarray, torch.Tensor]:
    is_torch = isinstance(volume, torch.Tensor)
    
    if is_torch:
        volume = volume.float()
        air_mask = volume <= -800
        bone_mask = volume > 350
        
        density = volume.clone()
        
        if torch.any(~air_mask & ~bone_mask):
            min_soft_tissue = torch.min(volume[~air_mask & ~bone_mask])
        else:
            min_soft_tissue = torch.min(volume)
        
        density[air_mask] = min_soft_tissue
        density[bone_mask] *= bone_attenuation_multiplier
        
        density_min = torch.min(density)
        density_max = torch.max(density)
        density = (density - density_min) / (density_max - density_min + 1e-8)
    else:
        volume = volume.astype(np.float32)
        air_mask = volume <= -800
        bone_mask = volume > 350
        
        density = volume.copy()
        
        soft_tissue_mask = ~air_mask & ~bone_mask
        if np.any(soft_tissue_mask):
            min_soft_tissue = np.min(volume[soft_tissue_mask])
        else:
            min_soft_tissue = np.min(volume)
        
        density[air_mask] = min_soft_tissue
        density[bone_mask] *= bone_attenuation_multiplier
        
        density_min = np.min(density)
        density_max = np.max(density)
        density = (density - density_min) / (density_max - density_min + 1e-8)
    return density

def normalize_xrays_to_match(
    generated_xrays: torch.Tensor,
    reference_xrays: torch.Tensor
) -> torch.Tensor:
    if generated_xrays.dim() == 4:
        generated_xrays = generated_xrays.squeeze(1)
    if reference_xrays.dim() == 4:
        reference_xrays = reference_xrays.squeeze(1)
    
    num_views = generated_xrays.shape[0]
    normalized_views = []
    
    for v in range(num_views):
        gen_view = generated_xrays[v]
        ref_view = reference_xrays[v] if v < reference_xrays.shape[0] else reference_xrays[0]
        
        ref_min = ref_view.min()
        ref_max = ref_view.max()
        ref_range = ref_max - ref_min if (ref_max - ref_min) > 1e-8 else 1.0
        
        gen_min = gen_view.min()
        gen_max = gen_view.max()
        gen_range = gen_max - gen_min
        
        if gen_range > 1e-8:
            gen_normalized = (gen_view - gen_min) / gen_range
            gen_scaled = gen_normalized * ref_range + ref_min
        else:
            gen_scaled = torch.full_like(gen_view, ref_view.mean())
        
        normalized_views.append(gen_scaled)
    
    return torch.stack(normalized_views, dim=0)

def get_residual(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return estimate - reference

def get_centered_affine(
    spacing: Union[np.ndarray, torch.Tensor, List, Tuple],
    volume_shape_xyz: Union[np.ndarray, torch.Tensor, List, Tuple]
) -> Union[np.ndarray, torch.Tensor]:
    is_torch = isinstance(spacing, torch.Tensor) or isinstance(volume_shape_xyz, torch.Tensor)
    
    if is_torch:
        if not isinstance(spacing, torch.Tensor):
            spacing = torch.tensor(spacing, dtype=torch.float32)
        else:
            spacing = spacing.float()
        
        if not isinstance(volume_shape_xyz, torch.Tensor):
            volume_shape_xyz = torch.tensor(volume_shape_xyz, dtype=torch.float32)
        else:
            volume_shape_xyz = volume_shape_xyz.float()
        
        affine = torch.eye(4, dtype=spacing.dtype, device=spacing.device)
        scaling = torch.diag(spacing)
        affine[:3, :3] = scaling
        
        physical_size = volume_shape_xyz * spacing
        centered_origin = -physical_size / 2.0
        affine[:3, 3] = centered_origin
    else:
        spacing = np.array(spacing, dtype=np.float32)
        volume_shape_xyz = np.array(volume_shape_xyz, dtype=np.float32)
        
        affine = np.eye(4, dtype=np.float32)
        scaling = np.diag(spacing)
        affine[:3, :3] = scaling
        
        physical_size = volume_shape_xyz * spacing
        centered_origin = -physical_size / 2.0
        affine[:3, 3] = centered_origin
    return affine

def euler_to_rotation_matrix(pitch: float, roll: float, yaw: float) -> np.ndarray:
    p = pitch * math.pi / 180.0
    r = roll * math.pi / 180.0
    y = yaw * math.pi / 180.0

    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(p), -math.sin(p)],
        [0, math.sin(p), math.cos(p)]
    ], dtype=np.float32)

    Ry = np.array([
        [math.cos(r), 0, math.sin(r)],
        [0, 1, 0],
        [-math.sin(r), 0, math.cos(r)]
    ], dtype=np.float32)

    Rz = np.array([
        [math.cos(y), -math.sin(y), 0],
        [math.sin(y), math.cos(y), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    return Rz @ Ry @ Rx

def gen_rotation_translate(device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    rotations = []
    
    for angle in [30, 90, 150]:
        rotations.append([0.0, 0.0, float(angle)])
    
    for angle in [30, 90, 150]:
        rotations.append([float(angle), 0.0, 0.0])
        if angle > 0:
            rotations.append([-float(angle), 0.0, 0.0])
    
    for angle in [0, 60, 120]:
        rotations.append([0.0, float(angle), 0.0])
        if angle > 0:
            rotations.append([0.0, -float(angle), 0.0])
    
    for angle in [30, 60]:
        rotations.append([float(angle), float(angle), 0.0])
        rotations.append([float(angle), -float(angle), 0.0])
        rotations.append([float(angle), 0.0, float(angle)])
        rotations.append([float(angle), 0.0, -float(angle)])
        rotations.append([0.0, float(angle), float(angle)])
        rotations.append([0.0, float(angle), -float(angle)])
    
    rotations_tensor = torch.tensor(rotations, device=device, dtype=torch.float32)
    translations_tensor = torch.tensor([[0.0, 850.0, 0.0]], device=device)
    
    return rotations_tensor, translations_tensor

def initialize_paths(args: Any, config: Dict[str, Any], device: str) -> List[str]:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_base = os.path.join(args.output_dir, timestamp)
    debug_dir = os.path.join(output_base, "debug")
    samples_dir = os.path.join(output_base, "samples")
    conditions_dir = os.path.join(output_base, "conditions")
    xray_verification_dir = os.path.join(output_base, "xray_verification")
    
    for d in [output_base, debug_dir, samples_dir, conditions_dir, xray_verification_dir]:
        os.makedirs(d, exist_ok=True)
        
    log_level = logging.INFO if getattr(args, "verbose", False) else logging.ERROR
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_base, 'generation.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    
    with open(os.path.join(output_base, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    logger.info(f"Starting generation with device: {device}")
    logger.info(f"Output directory structure created at: {output_base}")
    logger.info(f"Debug directory: {debug_dir}")
    logger.info(f"Samples directory: {samples_dir}")
    logger.info(f"Conditions directory: {conditions_dir}")
    
    return [output_base, debug_dir, samples_dir, conditions_dir, xray_verification_dir, logger]

def save_middle_slices(
    volume: torch.Tensor, 
    step: str, 
    save_dir: str = "intermediate_results", 
    ref_volume: Optional[torch.Tensor] = None
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    
    B, C, D, H, W = volume.shape
    volume_np = volume[0, 0].cpu().to(torch.float32).detach().numpy()
    
    est_sagittal = volume_np[:, :, W//2]
    est_coronal = volume_np[:, H//2, :]
    est_axial = volume_np[D//2, :, :]
    
    slices = [est_sagittal, est_coronal, est_axial]
    normalized_slices = []
    for slice_data in slices:
        min_val, max_val = slice_data.min(), slice_data.max()
        if max_val > min_val:
            normalized_slices.append((slice_data - min_val) / (max_val - min_val))
        else:
            normalized_slices.append(slice_data)
    
    est_sagittal, est_coronal, est_axial = normalized_slices
    
    if ref_volume is not None:
        if len(ref_volume.shape) == 3:
            ref_np = ref_volume.cpu().to(torch.float32).detach().numpy()
        elif len(ref_volume.shape) == 5:
            ref_np = ref_volume[0, 0].cpu().to(torch.float32).detach().numpy()
        else:
            raise ValueError(f"Unexpected reference volume shape: {ref_volume.shape}")
        
        ref_sagittal = ref_np[:, :, W//2]
        ref_coronal = ref_np[:, H//2, :]
        ref_axial = ref_np[D//2, :, :]
        
        ref_slices = [ref_sagittal, ref_coronal, ref_axial]
        normalized_ref_slices = []
        for slice_data in ref_slices:
            min_val, max_val = slice_data.min(), slice_data.max()
            if max_val > min_val:
                normalized_ref_slices.append((slice_data - min_val) / (max_val - min_val))
            else:
                normalized_ref_slices.append(slice_data)
        
        ref_sagittal, ref_coronal, ref_axial = normalized_ref_slices
        
        diff_sagittal = np.abs(est_sagittal - ref_sagittal)
        diff_coronal = np.abs(est_coronal - ref_coronal)
        diff_axial = np.abs(est_axial - ref_axial)
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        axes[0,0].imshow(est_sagittal, cmap='gray')
        axes[0,0].set_title(f'Estimated Sagittal (t={step})')
        axes[0,1].imshow(ref_sagittal, cmap='gray')
        axes[0,1].set_title('Reference Sagittal')
        im = axes[0,2].imshow(diff_sagittal, cmap='hot')
        axes[0,2].set_title('Difference')
        plt.colorbar(im, ax=axes[0,2])
        
        axes[1,0].imshow(est_coronal, cmap='gray')
        axes[1,0].set_title(f'Estimated Coronal (t={step})')
        axes[1,1].imshow(ref_coronal, cmap='gray')
        axes[1,1].set_title('Reference Coronal')
        im = axes[1,2].imshow(diff_coronal, cmap='hot')
        axes[1,2].set_title('Difference')
        plt.colorbar(im, ax=axes[1,2])
        
        axes[2,0].imshow(est_axial, cmap='gray')
        axes[2,0].set_title(f'Estimated Axial (t={step})')
        axes[2,1].imshow(ref_axial, cmap='gray')
        axes[2,1].set_title('Reference Axial')
        im = axes[2,2].imshow(diff_axial, cmap='hot')
        axes[2,2].set_title('Difference')
        plt.colorbar(im, ax=axes[2,2])
        
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(est_sagittal, cmap='gray')
        axes[0].set_title(f'Estimated Sagittal (t={step})')
        axes[1].imshow(est_coronal, cmap='gray')
        axes[1].set_title(f'Estimated Coronal (t={step})')
        axes[2].imshow(est_axial, cmap='gray')
        axes[2].set_title(f'Estimated Axial (t={step})')
    
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'volume_comparison_step_{step}.png'))
    plt.close()

def save_comparison_grid(
    projs: torch.Tensor, 
    refs: torch.Tensor, 
    step: str, 
    save_dir: str = "intermediate_results"
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    
    n_angles = projs.shape[0]
    
    def infer_image_shape(proj_shape, ref_shape):
        shapes = [proj_shape, ref_shape]
        for shape in shapes:
            if len(shape) >= 3:
                if shape[1] >= 2 and shape[2] >= 2:
                    return (shape[1], shape[2])
            elif len(shape) == 2:
                if shape[0] == n_angles:
                    total = shape[1]
                    side = int(np.sqrt(total))
                    if side * side == total:
                        return (side, side)
        return (128, 128)
    
    proj_shape = tuple(projs.shape)
    ref_shape = tuple(refs.shape)
    expected_h, expected_w = infer_image_shape(proj_shape, ref_shape)
    
    fig = plt.figure(figsize=(max(4 * n_angles, 10), 12))
    gs = plt.GridSpec(3, n_angles, figure=fig)
    
    def normalize_batch(tensor: torch.Tensor) -> np.ndarray:
        np_array = tensor.cpu().to(torch.float32).detach().numpy()
        original_shape = np_array.shape
        
        if np_array.ndim == 4:
            if np_array.shape[1] == 1:
                np_array = np_array.squeeze(1)
            else:
                np_array = np_array[:, 0, :, :]
        
        if np_array.ndim == 3:
            if np_array.shape[1] >= 2 and np_array.shape[2] >= 2:
                if np_array.shape[0] != n_angles:
                    if np_array.shape[0] == 1:
                        np_array = np.repeat(np_array, n_angles, axis=0)
                    else:
                        np_array = np_array[:n_angles, :, :]
            else:
                if np_array.shape[1] == 1 and np_array.shape[2] == expected_h * expected_w:
                    np_array = np_array.reshape(np_array.shape[0], expected_h, expected_w)
                elif np_array.shape[1] == 1:
                    total = np_array.shape[2]
                    side = int(np.sqrt(total))
                    if side * side == total:
                        np_array = np_array.reshape(np_array.shape[0], side, side)
                elif np_array.shape[2] == 1:
                    total = np_array.shape[1]
                    side = int(np.sqrt(total))
                    if side * side == total:
                        np_array = np_array.reshape(np_array.shape[0], side, side)
        
        if np_array.ndim == 2:
            if np_array.shape[0] == n_angles:
                total_pixels = np_array.shape[1]
                expected_pixels = expected_h * expected_w
                if total_pixels == expected_pixels:
                    np_array = np_array.reshape(n_angles, expected_h, expected_w)
                else:
                    side_len = int(np.sqrt(total_pixels))
                    if side_len * side_len == total_pixels:
                        np_array = np_array.reshape(n_angles, side_len, side_len)
                    else:
                        np_array = np_array[np.newaxis, :, :]
                        if n_angles > 1:
                            np_array = np.repeat(np_array, n_angles, axis=0)
            else:
                np_array = np_array[np.newaxis, :, :]
                if n_angles > 1:
                    np_array = np.repeat(np_array, n_angles, axis=0)
        
        if np_array.ndim == 1:
            total_pixels = np_array.shape[0]
            side_len = int(np.sqrt(total_pixels))
            if side_len * side_len == total_pixels:
                img = np_array.reshape(side_len, side_len)
                np_array = img[np.newaxis, :, :]
                if n_angles > 1:
                    np_array = np.repeat(np_array, n_angles, axis=0)
            else:
                raise ValueError(f"Cannot reshape 1D array of size {total_pixels} to 2D square image. Original shape: {original_shape}")
        
        if np_array.ndim != 3:
            raise ValueError(f"Failed to convert to 3D array. Original shape: {original_shape}, current shape: {np_array.shape}")
        
        if np_array.shape[0] != n_angles:
            raise ValueError(f"First dimension {np_array.shape[0]} doesn't match n_angles {n_angles}. Original shape: {original_shape}")
        
        if np_array.shape[1] < 2 or np_array.shape[2] < 2:
            raise ValueError(f"Image dimensions too small: {np_array.shape[1]}x{np_array.shape[2]}. Original shape: {original_shape}")
        
        min_vals = np_array.min(axis=(1, 2), keepdims=True)
        max_vals = np_array.max(axis=(1, 2), keepdims=True)
        normalized = (np_array - min_vals) / (max_vals - min_vals + 1e-8)
        
        return normalized
    
    proj_nps = normalize_batch(projs)
    ref_nps = normalize_batch(refs)
    diffs = np.abs(proj_nps - ref_nps)
    
    vmin_ref, vmax_ref = ref_nps.min(), ref_nps.max()
    vmin_proj, vmax_proj = proj_nps.min(), proj_nps.max()
    vmin_diff, vmax_diff = diffs.min(), diffs.max()
    
    for angle_idx in range(n_angles):
        ax = fig.add_subplot(gs[0, angle_idx])
        im = ax.imshow(ref_nps[angle_idx], cmap='gray', vmin=vmin_ref, vmax=vmax_ref)
        if angle_idx == n_angles - 1:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f'Angle {angle_idx}')
        ax.axis('off')
        if angle_idx == 0:
            ax.set_ylabel('Reference', fontsize=12)
        
        ax = fig.add_subplot(gs[1, angle_idx])
        im = ax.imshow(proj_nps[angle_idx], cmap='gray', vmin=vmin_proj, vmax=vmax_proj)
        if angle_idx == n_angles - 1:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
        if angle_idx == 0:
            ax.set_ylabel('Estimated', fontsize=12)
        
        ax = fig.add_subplot(gs[2, angle_idx])
        im = ax.imshow(diffs[angle_idx], cmap='hot', vmin=vmin_diff, vmax=vmax_diff)
        if angle_idx == n_angles - 1:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
        if angle_idx == 0:
            ax.set_ylabel('Difference', fontsize=12)
    
    plt.suptitle(f'Step {step}', y=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'comparison_step_{step}.png'), dpi=150, bbox_inches='tight')
    plt.close()

def saving_conditions(path: str, cond: torch.Tensor, names: str) -> None:
    for i in range(cond.shape[0]):
        array = cond[i, 0].cpu().numpy()
        plt.imshow(array)
        plt.axis('off')
        plt.savefig(os.path.join(path, f"{names}_channel_{i}.png"), bbox_inches='tight', pad_inches=0)
        plt.close()

def save_xrays(
    xrays: Union[torch.Tensor, np.ndarray],
    save_dir: str,
    prefix: str = "xray",
    save_individual: bool = True,
    save_grid: bool = True,
    grid_cols: Optional[int] = None
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    
    is_torch = isinstance(xrays, torch.Tensor)
    
    if is_torch:
        xrays_np = xrays.detach().cpu().numpy()
    else:
        xrays_np = np.array(xrays)
    
    original_shape = xrays_np.shape
    
    if xrays_np.ndim == 4:
        B, C, H, W = xrays_np.shape
        if C == 1:
            xrays_np = xrays_np.squeeze(1)
        else:
            xrays_np = xrays_np[:, 0, :, :]
    elif xrays_np.ndim == 3:
        if xrays_np.shape[0] == 1 and xrays_np.shape[1] > 1 and xrays_np.shape[2] > 1:
            xrays_np = xrays_np.squeeze(0)
        elif xrays_np.shape[1] == 1:
            xrays_np = xrays_np.squeeze(1)
    elif xrays_np.ndim == 2:
        xrays_np = xrays_np[np.newaxis, :, :]
    
    if xrays_np.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array after processing, got {xrays_np.ndim}D with shape {original_shape}")
    
    num_xrays = xrays_np.shape[0]
    H, W = xrays_np.shape[1], xrays_np.shape[2]
    
    xray_images = []
    
    for i in range(num_xrays):
        xray = xrays_np[i]
        
        xray_min = xray.min()
        xray_max = xray.max()
        if xray_max > xray_min:
            xray_normalized = (xray - xray_min) / (xray_max - xray_min)
        else:
            xray_normalized = xray
        
        xray_uint8 = (xray_normalized * 255).astype(np.uint8)
        xray_images.append(xray_uint8)
        
        if save_individual:
            cv2.imwrite(os.path.join(save_dir, f"{prefix}_{i:03d}.png"), xray_uint8)
    
    if save_grid and num_xrays > 1:
        if grid_cols is None:
            grid_cols = int(np.ceil(np.sqrt(num_xrays)))
        grid_rows = int(np.ceil(num_xrays / grid_cols))
        
        grid_image = np.zeros((grid_rows * H, grid_cols * W), dtype=np.uint8)
        
        for idx in range(num_xrays):
            row = idx // grid_cols
            col = idx % grid_cols
            grid_image[row*H:(row+1)*H, col*W:(col+1)*W] = xray_images[idx]
        
        cv2.imwrite(os.path.join(save_dir, f"{prefix}_grid.png"), grid_image)

def create_val_loader(config: Dict[str, Any], data_type="val") -> Any:
    to_3tuple = _ntuple(3)
    repo_root = Path(__file__).resolve().parents[1]

    def resolve_split_path(path: str) -> str:
        split_path = Path(path).expanduser()
        if not split_path.is_absolute():
            split_path = repo_root / split_path
        return str(split_path.resolve())

    train_txt = resolve_split_path(config.data.train_txt)
    test_txt = resolve_split_path(config.data.test_txt)
    logger.info(f"Creating validation dataset loader for task: {config.data.task}")
    logger.info(f"Creating validation dataset loader for path: {config.data.path}")
    logger.info(f"Creating validation dataset loader for image size: {config.data.image_size}")
    logger.info(f"Creating validation dataset loader for seed: {config.training.seed}")
    logger.info(f"Creating validation dataset loader for augment: {config.data.augment}")
    logger.info(f"Using training split list: {train_txt}")
    logger.info(f"Using test split list: {test_txt}")
    return get_voxel_dataset(
        config.data.path,
        task=config.data.task,
        config=config,
        roi_size=to_3tuple(config.data.image_size),
        data_type=data_type,
        train_txt=train_txt,
        test_txt=test_txt,
        seed=config.training.seed,
        augment=False,
    )

def create_sample_subject_from_drr(dataset: Any, debug_dir: str, logger: logging.Logger) -> Any:
    random_idx = random.randint(0, len(dataset)-1)
    sample_subj = saving_sample_to_extract_instrins(
        image=dataset[random_idx]["image"],
        affine=np.eye(4),
        save_path=os.path.join(debug_dir, "sample_template.nii.gz")
    )    
    logger.info(f"Created template subject from index: {random_idx}")
    logger.info(f"Template volume shape: {sample_subj.volume.shape}")
    logger.info(f"Template spacing: {sample_subj.volume.spacing}")
    return sample_subj

def get_condition_from_dataset(dataset: Any) -> torch.Tensor:
    outputs = []
    for idx in tqdm(range(len(dataset)), desc="Processing dataset"):
        p_x = torch.sum(dataset[idx], dim=1)
        p_y = torch.sum(dataset[idx], dim=2)
        p_z = torch.sum(dataset[idx], dim=3)
        outputs.append(torch.stack([p_x, p_y, p_z], dim=-1))
    
    return torch.cat(outputs, dim=0)

def compute_volume_metrics(reference_volume: np.ndarray, generated_volume: np.ndarray) -> Dict[str, float]:
    ref_vol = (reference_volume - reference_volume.min()) / (reference_volume.max() - reference_volume.min())
    gen_vol = (generated_volume - generated_volume.min()) / (generated_volume.max() - generated_volume.min())
    
    mse = np.mean((ref_vol - gen_vol) ** 2)
    psnr_val = psnr(ref_vol, gen_vol, data_range=1.0)
    ssim_val = ssim(ref_vol, gen_vol, data_range=1.0, channel_axis=None)
    
    signal_power = np.mean(ref_vol ** 2)
    noise_power = mse
    eps = 1e-10
    snr_val = 10 * np.log10(signal_power / (noise_power + eps)) if noise_power > 0 else 100.0
    
    return {
        'mse': float(mse),
        'psnr': float(psnr_val),
        'ssim': float(ssim_val),
        'snr': float(snr_val),
        'signal_power': float(signal_power),
        'noise_power': float(noise_power)
    }

if __name__ == "__main__":
    pass
