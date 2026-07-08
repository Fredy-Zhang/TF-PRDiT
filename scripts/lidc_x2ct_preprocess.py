#!/usr/bin/env python3
"""
Preprocess CT volumes like the reference dataset pipeline (axis-wise resizing only).

What this does (to match the reference):
- Keeps data handling only; NO saving, NO normalization/quantile/clipping.
- Resizes the 3D array along one axis at a time:
    * axial   volume:   (trg_ct_res, base_ct_res, base_ct_res)
    * coronal volume:   (base_ct_res, trg_ct_res, base_ct_res)
    * sagittal volume:  (base_ct_res, base_ct_res, trg_ct_res)
- Returns NumPy arrays, plus convenience stacks of 2D slices per view.

Axis convention:
- The reference treats volume as (z, x, y). NIfTI is commonly (z, y, x).
  We swap axes 1 and 2 when loading to maintain the reference’s (z, x, y).
"""

from __future__ import annotations
from typing import Dict, Tuple, List
import argparse
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage


# ---------- Utils ----------

def _is_array_like(obj) -> bool:
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class ResizeImage:
    """
    Resize a 3D array to target size (z, x, y) using linear interpolation (order=1),
    matching the reference behavior.

    Example:
        resizer = ResizeImage(size=(128, 320, 320))
        out = resizer(vol)  # vol assumed (z, x, y)
    """
    def __init__(self, size=(3, 256, 256)):
        if not _is_array_like(size):
            raise ValueError("each dimension of size must be defined")
        self.size = np.array(size, dtype=np.float32)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if img.ndim != 3:
            raise ValueError(f"Expected a 3D array (z, x, y), got shape {img.shape}")
        z, x, y = img.shape
        ori_shape = np.array((z, x, y), dtype=np.float32)
        zoom_factors = self.size / ori_shape
        # ndimage.interpolation.zoom is deprecated; use ndimage.zoom
        return ndimage.zoom(img, zoom_factors, order=1)


# ---------- Core loading & resizing (reference-parity) ----------

def load_nifti_as_reference_array(nifti_path: str) -> np.ndarray:
    """
    Load NIfTI and return np.ndarray in (z, x, y) to match the reference convention.
    NIfTI is typically (z, y, x) -> swap axes 1 and 2.
    """
    img = nib.load(nifti_path)
    vol = np.asarray(img.get_fdata(), dtype=np.float32)  # commonly (z, y, x)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI volume, got shape {vol.shape}")
    # Convert to (z, x, y) to mirror the reference code
    vol = vol.swapaxes(1, 2)
    return vol


def make_axis_resized_volumes(
    ori_ct: np.ndarray,
    trg_ct_res: int,
    base_ct_res: int = 320
) -> Dict[str, np.ndarray]:
    """
    Create three volumes like the reference:
        - axial   : resize axis-0 to trg (z), others to base -> (trg, base, base)
        - coronal : resize axis-1 to trg (x), others to base -> (base, trg, base)
        - sagittal: resize axis-2 to trg (y), others to base -> (base, base, trg)
    """
    base_vec = [base_ct_res, base_ct_res, base_ct_res]

    # Axial volume
    sz = base_vec.copy(); sz[0] = trg_ct_res
    axial_vol = ResizeImage(size=sz)(ori_ct)

    # Coronal volume
    sz = base_vec.copy(); sz[1] = trg_ct_res
    coronal_vol = ResizeImage(size=sz)(ori_ct)

    # Sagittal volume
    sz = base_vec.copy(); sz[2] = trg_ct_res
    sagittal_vol = ResizeImage(size=sz)(ori_ct)

    return {
        "axial_volume": axial_vol,         # (trg, base, base)
        "coronal_volume": coronal_vol,     # (base, trg, base)
        "sagittal_volume": sagittal_vol    # (base, base, trg)
    }


def stack_slices(volumes: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Produce stacks of (N, H, W) 2D slices for each view (N == trg_ct_res):
      - axial   slices: axial_volume[i]        -> (base, base)
      - coronal slices: coronal_volume[:, i]   -> (base, base) -> transpose to (N, H, W)
      - sagittal slices: sagittal_volume[..., i]-> (base, base) -> transpose to (N, H, W)
    """
    axial = volumes["axial_volume"]          # (N, H, W)
    coronal = volumes["coronal_volume"]      # (H, N, W)
    sagittal = volumes["sagittal_volume"]    # (H, W, N)

    axial_slices = axial.copy()                              # already (N, H, W)
    coronal_slices = np.transpose(coronal, (1, 0, 2))        # (N, H, W)
    sagittal_slices = np.transpose(sagittal, (2, 0, 1))      # (N, H, W)

    # All three stacks end up (trg_ct_res, base_ct_res, base_ct_res)
    return {
        "axial_slices": axial_slices,
        "coronal_slices": coronal_slices,
        "sagittal_slices": sagittal_slices
    }


def preprocess_nifti_like_reference(
    nifti_path: str,
    trg_ct_res: int,
    base_ct_res: int = 320
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Full pipeline (NO saving, NO normalization/clipping):
      1) Load NIfTI -> np.ndarray (z, x, y) to match reference convention.
      2) Build three axis-resized volumes like the reference.
      3) Provide stacks of 2D slices for each orientation.

    Returns:
      volumes: dict with keys {'axial_volume','coronal_volume','sagittal_volume'}
               shapes: (trg, base, base), (base, trg, base), (base, base, trg)
      slices:  dict with keys {'axial_slices','coronal_slices','sagittal_slices'}
               each shape: (trg, base, base)
    """
    ori_ct = load_nifti_as_reference_array(nifti_path)  # (z, x, y)
    volumes = make_axis_resized_volumes(ori_ct, trg_ct_res, base_ct_res)
    slices = stack_slices(volumes)
    return volumes, slices


# ---------- Optional: CLI to quickly inspect shapes (no writing) ----------

def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Axis-wise CT resizing like the reference (no saving)."
    )
    parser.add_argument("--nifti", required=True, help="Path to input NIfTI (e.g., ct.nii.gz)")
    parser.add_argument("--target_res", type=int, default=128, help="Target slices per axis (trg_ct_res)")
    parser.add_argument("--base_res", type=int, default=320, help="Base resolution for non-target axes")
    args = parser.parse_args(argv)

    volumes, slices = preprocess_nifti_like_reference(
        nifti_path=args.nifti,
        trg_ct_res=args.target_res,
        base_ct_res=args.base_res
    )

    # Print shapes for sanity check
    print("=== Volume shapes (Z, X, Y) ===")
    for k, v in volumes.items():
        print(f"{k:16s}: {tuple(v.shape)}")
    print("\n=== Slice stacks (N, H, W) ===")
    for k, v in slices.items():
        print(f"{k:16s}: {tuple(v.shape)}")


if __name__ == "__main__":
    main()
