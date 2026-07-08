"""
Script for preprocessing the LIDC-IDRI dataset.

Keeps: DICOM -> NIfTI conversion (dicom2nifti)
Optimizes: NIfTI preprocessing to match the 'reference' behavior
"""

import argparse
import os
import shutil
import glob
import random
import dicom2nifti
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom as ndzoom


# ------------------------- helpers ------------------------- #

def resample_isotropic_1mm(vol_xyz: np.ndarray, spacing_xyz, order: int = 1):
    """
    Resample (X,Y,Z) volume to 1.0mm isotropic spacing.
    spacing_xyz is (sx, sy, sz) in mm.
    """
    spacing_xyz = np.asarray(spacing_xyz, dtype=np.float32)
    target = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    zoom_factors = spacing_xyz / target
    vol = ndzoom(vol_xyz, zoom=zoom_factors, order=order, mode='nearest')
    return vol, target


def crop_pad_to_standard(vol_xyz: np.ndarray, scale: int, pad_value: float = 0.0):
    """
    Make a fixed cube (scale x scale x scale) with the reference rules:
      - Z: if longer -> keep LAST 'scale' slices; else symmetric pad
      - Y/X: center crop to 'scale' or symmetric pad to 'scale'
    Shapes use (X, Y, Z).
    """
    x, y, z = vol_xyz.shape

    # Z: keep last 'scale' or symmetric pad
    if z >= scale:
        vol_xyz = vol_xyz[:, :, z - scale:z]
    else:
        z_before = (scale - z) // 2
        z_after = scale - z - z_before
        vol_xyz = np.pad(vol_xyz, ((0, 0), (0, 0), (z_before, z_after)),
                         mode='constant', constant_values=pad_value)

    # Y: center crop/pad
    x, y, z = vol_xyz.shape
    if y >= scale:
        y0 = (y - scale) // 2
        vol_xyz = vol_xyz[:, y0:y0 + scale, :]
    else:
        y_before = (scale - y) // 2
        y_after = scale - y - y_before
        vol_xyz = np.pad(vol_xyz, ((0, 0), (y_before, y_after), (0, 0)),
                         mode='constant', constant_values=pad_value)

    # X: center crop/pad
    x, y, z = vol_xyz.shape
    if x >= scale:
        x0 = (x - scale) // 2
        vol_xyz = vol_xyz[x0:x0 + scale, :, :]
    else:
        x_before = (scale - x) // 2
        x_after = scale - x - x_before
        vol_xyz = np.pad(vol_xyz, ((x_before, x_after), (0, 0), (0, 0)),
                         mode='constant', constant_values=pad_value)

    return vol_xyz


def resize_to_target(vol_xyz: np.ndarray, target_size: int, order: int = 1):
    """
    Resize (X,Y,Z) cube to (target_size, target_size, target_size).
    """
    if vol_xyz.shape == (target_size, target_size, target_size):
        return vol_xyz
    factors = np.array([target_size / s for s in vol_xyz.shape], dtype=np.float32)
    return ndzoom(vol_xyz, zoom=factors, order=order, mode='nearest')


def load_nifti_xyz(path: str):
    """
    Load NIfTI as (X,Y,Z) float32 and return (data, affine, spacing_xyz, header).
    """
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    spacing_xyz = np.array(img.header.get_zooms()[:3], dtype=np.float32)
    return data, img.affine, spacing_xyz, img.header


def save_nifti_xyz(data_xyz: np.ndarray, affine, spacing_xyz, out_path: str, like_header=None):
    """
    Save NIfTI with updated spacing.
    """
    if like_header is not None:
        header = like_header.copy()
        header.set_zooms(tuple(spacing_xyz.tolist()))
        img = nib.Nifti1Image(data_xyz.astype(np.float32), affine, header=header)
    else:
        img = nib.Nifti1Image(data_xyz.astype(np.float32), affine)
        img.header.set_zooms(tuple(spacing_xyz.tolist()))
    nib.save(img, out_path)


# ------------------------- core preprocessing ------------------------- #

def preprocess_nifti(
    input_path: str,
    output_path: str,
    trg_ct_res: int = 128,
    base_ct_res: int = 320,
    hu_clip=None,                    # e.g. (-1000, 1000) or None
    normalize: bool = True,
    mask_path: str = None            # optional body mask NIfTI to zero-out table/air
):
    """
    Reference-matching preprocessing:
      1) Load NIfTI (X,Y,Z)
      2) (Optional) HU clip
      3) Resample to 1mm isotropic
      4) Find minimum HU value for adaptive padding (lowest tissue value)
      5) (Optional) Apply body mask -> set outside to adaptive padding value
      5b) Crop/pad to base cube with adaptive padding (uses minimum HU value)
      6) (Optional) normalize to [0,1] (tissue-only statistics)
      7) Resize to target cube (e.g., 128^3)
      8) Save with spacing set to (1,1,1)
    """
    print(f'Process image: {input_path}')
    vol_xyz, affine, spacing_xyz, header = load_nifti_xyz(input_path)

    # Step 2: HU clip
    if hu_clip is not None:
        lo, hi = hu_clip
        vol_xyz = np.clip(vol_xyz, lo, hi)

    # Step 3: resample to 1mm
    vol_xyz, new_spacing = resample_isotropic_1mm(vol_xyz, spacing_xyz, order=1)
    print(f"[Resample] -> shape: {vol_xyz.shape}")

    # Step 4: Find minimum HU value BEFORE masking (to use for adaptive padding)
    tissue_min = float(vol_xyz.min())
    print(f"[4] Tissue min HU value (before masking): {tissue_min}")
    adaptive_pad_value = tissue_min

    # Step 5: optional mask (use adaptive padding value for masked regions)
    if mask_path and os.path.exists(mask_path):
        mask_xyz, _, mask_spacing, _ = load_nifti_xyz(mask_path)
        if np.any(mask_spacing != 1.0):
            # bring mask to 1mm with nearest neighbor
            zf = mask_spacing / np.array([1.0, 1.0, 1.0], dtype=np.float32)
            mask_xyz = ndzoom(mask_xyz, zoom=zf, order=0, mode='nearest')
        # align shapes (in case of 1-voxel mismatches)
        sx = min(vol_xyz.shape[0], mask_xyz.shape[0])
        sy = min(vol_xyz.shape[1], mask_xyz.shape[1])
        sz = min(vol_xyz.shape[2], mask_xyz.shape[2])
        vol_xyz = vol_xyz[:sx, :sy, :sz]
        mask_xyz = mask_xyz[:sx, :sy, :sz]
        # Use adaptive padding value instead of 0.0
        vol_xyz[mask_xyz <= 0] = adaptive_pad_value
        print(f"[5] Mask applied (filled with {adaptive_pad_value:.2f}) -> shape: {vol_xyz.shape}")

    # Step 5b: crop/pad to base cube with adaptive padding
    vol_xyz = crop_pad_to_standard(vol_xyz, scale=base_ct_res, pad_value=adaptive_pad_value)
    print(f"[5b] Standard cube {base_ct_res}^3 -> shape: {vol_xyz.shape}")

    # Step 6: normalize to [0,1] (excluding padding areas from min/max calculation)
    if normalize:
        print("[6] Clip all values below -1000 ...")
        vol_xyz[vol_xyz < -1000] = -1000
        print("[4] Clip the upper quantile (0.999) to remove outliers ...")
        out_clipped = np.clip(vol_xyz, -1000, np.quantile(vol_xyz, 0.999))
        print("[5] Normalize the image ...")
        vol_xyz = (out_clipped - np.min(out_clipped)) / (np.max(out_clipped) - np.min(out_clipped))
        print(f'[6] After normalization - Min: {vol_xyz.min():.4f}, Max: {vol_xyz.max():.4f}')

    # Step 7: save base resolution version
    base_output_path = output_path.replace('.nii.gz', f'_{base_ct_res}.nii.gz')
    save_nifti_xyz(vol_xyz, affine, np.array([1.0, 1.0, 1.0], dtype=np.float32), base_output_path, like_header=header)
    print(f"[Save Base] {base_output_path} with shape {vol_xyz.shape}")

    # Step 8: resize to target cube
    if trg_ct_res is not None and trg_ct_res != base_ct_res:
        vol_xyz = resize_to_target(vol_xyz, target_size=trg_ct_res, order=1)
        print(f"[Resize] -> {trg_ct_res}^3 -> shape: {vol_xyz.shape}")

    # Step 9: save target resolution version
    target_output_path = output_path.replace('.nii.gz', f'_{trg_ct_res}.nii.gz')
    save_nifti_xyz(vol_xyz, affine, np.array([1.0, 1.0, 1.0], dtype=np.float32), target_output_path, like_header=header)
    print(f"[Save Target] {target_output_path} with shape {vol_xyz.shape}")
    print("-" * 79)


# ------------------------- CLI (kept DICOM->NIfTI) ------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dicom_dir', type=str, required=True,
                        help='Directory containing the original DICOM data')
    parser.add_argument('--nifti_dir', type=str, required=True,
                        help='Directory to store the converted & processed NIfTI files')
    parser.add_argument('--target_res', type=int, default=128,
                        help='Target cube size for final output (default: 128)')
    parser.add_argument('--base_res', type=int, default=320,
                        help='Base cube size before final resize (default: 320)')
    parser.add_argument('--delete_unprocessed', type=eval, default=True,
                        help='True to delete non-processed NIfTI files after processing')
    parser.add_argument('--mask_dir', type=str, default=None,
                        help='Optional directory of body masks aligned per case (NIfTI).')
    parser.add_argument('--clip', type=float, nargs=2, default=None, metavar=('LO', 'HI'),
                        help='Optional HU clipping, e.g. --clip -1000 1000')
    parser.add_argument('--no_normalize', action='store_true',
                        help='Disable [0,1] normalization')
    parser.add_argument('--debug_sample', action='store_true',
                        help='Debug mode: randomly process only 10 samples for testing')

    args = parser.parse_args()

    os.makedirs(args.nifti_dir, exist_ok=True)

    # ---------- Combined DICOM -> NIfTI -> Process workflow ----------
    # Get all directories in the DICOM directory
    all_items = [item for item in os.listdir(args.dicom_dir) 
                 if os.path.isdir(os.path.join(args.dicom_dir, item))]
    
    # Debug mode: randomly select 10 samples for testing
    if args.debug_sample:
        num_samples = min(10, len(all_items))
        selected_items = random.sample(all_items, num_samples)
        print(f'[DEBUG MODE] Total samples found: {len(all_items)}')
        print(f'[DEBUG MODE] Randomly selected {num_samples} samples for testing: {selected_items}')
        print('=' * 79)
    else:
        selected_items = all_items
        print(f'Processing all {len(all_items)} samples')
        print('=' * 79)
    
    for item in selected_items:
        item_path = os.path.join(args.dicom_dir, item)

        print(f'Converting and processing {item}')
        output_dir = os.path.join(args.nifti_dir, item)
        os.makedirs(output_dir, exist_ok=True)

        # Convert DICOM to NIfTI
        dicom2nifti.convert_directory(item_path, output_dir)

        # Find the generated NIfTI file
        nii_files = [f for f in os.listdir(output_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
        if not nii_files:
            print(f"Warning: No NIfTI file found after conversion for {item}")
            shutil.rmtree(item_path)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)
            continue

        # Process the NIfTI file immediately
        in_path = os.path.join(output_dir, nii_files[0])
        out_path = os.path.join(output_dir, 'processed.nii.gz')

        mask_path = None
        if args.mask_dir:
            # try to find a mask with the same folder name under mask_dir
            case_name = os.path.basename(output_dir)
            candidates = glob.glob(os.path.join(args.mask_dir, case_name, '*.nii*'))
            if candidates:
                mask_path = candidates[0]
                
        preprocess_nifti(
            input_path=in_path,
            output_path=out_path,
            trg_ct_res=args.target_res,
            base_ct_res=args.base_res,
            hu_clip=tuple(args.clip) if args.clip is not None else None,
            normalize=not args.no_normalize,
            mask_path=mask_path
        )
        print(f"Successfully processed {item}")

        # Delete the original NIfTI file after processing
        os.remove(in_path)

        # Remove the DICOM directory
        shutil.rmtree(item_path)
