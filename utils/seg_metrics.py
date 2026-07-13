"""Segmentation-based (task-based) evaluation of reconstructed CT volumes.

MSE / PSNR / SSIM say how close the voxels are. They do not say whether an
off-the-shelf clinical tool can still find the anatomy. This module answers the
second question by running **TotalSegmentator** on the reconstruction and on the
real CT *independently*, then comparing the two label maps.

Two families live here and they answer different questions. Report both:

  * ``structural_agreement`` -- Dice, **IoU**, HD95, ASSD and signed volume error
    between two *independent* segmentations. This is the task-based measure: it
    can say the structure is in the wrong place.
  * ``per_class_intensity`` -- **MAE**, RMSE and signed bias in HU, measured
    inside anatomy delineated on the *reference* volume. Both volumes see the
    same voxel set, so this is intensity fidelity and says nothing about
    location. Report ``bias_hu`` next to ``mae_hu``: it is signed, and it is what
    reveals systematic under-prediction (a blurred volume shows a strongly
    negative bone bias that the symmetric MAE hides completely).

Read every Dice against ``segment_ceiling`` -- see the warning on that function,
which is *not* the warning you may be expecting.

Data conventions this module assumes (they are easy to get wrong):
  * Array axis order is **(z, y, x)**, +x to the patient's left, +y posterior,
    +z inferior. RAS is therefore (-x, -y, -z), a left-handed frame, so the NIfTI
    affine has a negative determinant. That is valid; nnU-Net reorients from it.
  * Volumes are normalized to **[0, 1]**; ``to_hu`` maps them back.
  * The h5 ``spacing`` field is **stale** (it says 1.0 mm). The volumes are 256^3
    resamplings of a 320 mm field of view, so true spacing is 1.25 mm at 256^3
    and 2.50 mm at 128^3. ``spacing_for`` is the single source of truth. Confirmed
    against body extent: 1.25 mm gives a plausible ~250 mm AP thorax, 1.0 mm an
    implausible ~200 mm.
"""
import os
import shutil
import tempfile

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from utils.metrics import to_hu

# 256^3 crop of a 320 mm field of view -> 320/256 mm. See the module docstring.
SPACING_AT_256 = 1.25


# Anatomy is grouped into structures that (a) survive 2.5 mm sampling and (b) a
# reader recognises. Individual ribs and vertebrae are pooled: at this resolution
# a per-element Dice reflects labelling jitter, not reconstruction quality.
CLASS_GROUPS = {
    "lung": [
        "lung_upper_lobe_left", "lung_lower_lobe_left",
        "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
    ],
    "airway": ["trachea"],
    "heart": ["heart"],
    "great_vessels": ["aorta", "pulmonary_vein"],
    "bone": (
        [f"rib_left_{i}" for i in range(1, 13)]
        + [f"rib_right_{i}" for i in range(1, 13)]
        + [f"vertebrae_T{i}" for i in range(1, 13)]
        + ["vertebrae_C7", "vertebrae_L1", "vertebrae_L2"]
        + ["sternum", "scapula_left", "scapula_right",
           "clavicula_left", "clavicula_right"]
    ),
    "esophagus": ["esophagus"],
}

# Thin or small structures, where 2.5 mm voxels alone cost a lot of overlap. Kept
# in the tables but flagged, so a low number is not read as reconstruction failure.
LOW_RESOLUTION_CAVEAT = {"airway", "esophagus", "great_vessels"}


# ---------------------------------------------------------------------------
# Volume <-> NIfTI
# ---------------------------------------------------------------------------
def _affine(spacing):
    """Index-to-RAS affine for a (z, y, x) array transposed to (x, y, z).

    The arrays originate from SimpleITK: columns run to the patient's left, rows
    run posterior (DICOM LPS), slices run head-to-foot. That makes the frame
    (L, P, I) -- left-handed, hence the negative determinant. Valid NIfTI.
    """
    return np.diag([-spacing, -spacing, -spacing, 1.0])


def save_as_nifti(volume, path, spacing):
    """Write a normalized [0, 1] volume in (z, y, x) order as an HU NIfTI."""
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()
    volume = np.squeeze(np.asarray(volume))
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {volume.shape}")

    hu = np.round(to_hu(volume.astype(np.float64))).astype(np.int16)
    hu_xyz = np.transpose(hu, (2, 1, 0))
    nib.save(nib.Nifti1Image(hu_xyz, _affine(spacing)), path)
    return path


def spacing_for(volume_shape):
    """Isotropic mm spacing implied by an N^3 crop of the same physical extent."""
    n = volume_shape[-1]
    return SPACING_AT_256 * (256.0 / n)


def resize_volume(volume, size):
    """Trilinear resize of a 3D volume to (size, size, size)."""
    if isinstance(volume, np.ndarray):
        volume = torch.from_numpy(volume)
    volume = volume.squeeze().float()[None, None]
    out = F.interpolate(volume, size=(size,) * 3, mode="trilinear", align_corners=False)
    return out[0, 0]


# ---------------------------------------------------------------------------
# TotalSegmentator
# ---------------------------------------------------------------------------
def segment_labels(volume, spacing=None, device=None, fast=True, workdir=None):
    """Run TotalSegmentator, returning the raw label volume in (z, y, x) order.

    ``fast=True`` selects the 3 mm model, which is the right choice here: the
    1.5 mm model is trained for a resolution these volumes never had, and running
    it only invites the segmenter's own domain gap into the measurement.

    On a compute node with no egress, set ``TOTALSEG_HOME_DIR`` to a directory
    that already holds the weights, or the call will try to download them.
    """
    from totalsegmentator.python_api import totalsegmentator

    if spacing is None:
        spacing = spacing_for(np.squeeze(np.asarray(
            volume.cpu() if isinstance(volume, torch.Tensor) else volume)).shape)
    if device is None:
        device = "gpu" if torch.cuda.is_available() else "cpu"

    tmp = workdir or tempfile.mkdtemp(prefix="segmetrics_")
    try:
        in_path = save_as_nifti(volume, os.path.join(tmp, "ct.nii.gz"), spacing)
        out_path = os.path.join(tmp, "seg.nii.gz")
        totalsegmentator(in_path, out_path, task="total", fast=fast, ml=True,
                         device=device, quiet=True)

        # ml=True writes a single multilabel volume in the input geometry.
        seg_xyz = np.asarray(nib.load(out_path).dataobj).astype(np.uint8)
        return np.transpose(seg_xyz, (2, 1, 0))
    finally:
        if workdir is None:
            shutil.rmtree(tmp, ignore_errors=True)


def group_masks(seg):
    """Collapse a raw label volume into ``group -> boolean mask``."""
    from totalsegmentator.map_to_binary import class_map

    name_to_idx = {v: k for k, v in class_map["total"].items()}
    masks = {}
    for group, members in CLASS_GROUPS.items():
        m = np.zeros(seg.shape, dtype=bool)
        for name in members:
            idx = name_to_idx.get(name)
            if idx is not None:
                m |= seg == idx
        masks[group] = m
    return masks


def segment(volume, spacing=None, device=None, fast=True, workdir=None):
    """Segment a volume and return grouped boolean masks in (z, y, x) order."""
    return group_masks(segment_labels(volume, spacing, device, fast, workdir))


# ---------------------------------------------------------------------------
# Overlap and surface metrics
# ---------------------------------------------------------------------------
def _surface_distances(a, b, spacing):
    """Symmetric surface distances in mm between two boolean masks."""
    a_surf = a ^ ndimage.binary_erosion(a)
    b_surf = b ^ ndimage.binary_erosion(b)
    if not a_surf.any() or not b_surf.any():
        return None
    dt_b = ndimage.distance_transform_edt(~b_surf, sampling=spacing)
    dt_a = ndimage.distance_transform_edt(~a_surf, sampling=spacing)
    return np.concatenate([dt_b[a_surf], dt_a[b_surf]])


def _overlap(ref, pred, spacing):
    """Dice, IoU, surface distances and signed relative volume error.

    HD95 saturates at one voxel (2.50 mm on the 128^3 grid) as soon as the two
    surfaces are within a voxel of each other, which is exactly where good
    reconstructions sit -- so on near-perfect structures it carries no
    information. ``assd_mm`` averages rather than thresholding and still
    separates them. Keep both: HD95 regains its bite on poor reconstructions.
    """
    n_ref, n_pred = int(ref.sum()), int(pred.sum())
    inter = int((ref & pred).sum())
    union = n_ref + n_pred - inter

    out = {
        "dice": (2.0 * inter / (n_ref + n_pred)) if (n_ref + n_pred) else float("nan"),
        "iou": (inter / union) if union else float("nan"),
        "vol_ref_ml": n_ref * np.prod(spacing) / 1000.0,
        "vol_pred_ml": n_pred * np.prod(spacing) / 1000.0,
        "rel_vol_err": ((n_pred - n_ref) / n_ref) if n_ref else float("nan"),
    }
    # A missed structure has no surface, so its distances are NaN -- and a
    # downstream nanmean would then silently *drop* the model's worst failures,
    # flattering it. `detected` carries that fact so aggregation can report a
    # detection rate next to every mean. Note it only means both masks are
    # non-empty: Dice can still be 0.0 with detected=True if the predicted
    # structure is nowhere near the reference.
    out["detected"] = bool(n_ref and n_pred)
    d = _surface_distances(ref, pred, spacing) if out["detected"] else None
    out["hd95_mm"] = float(np.percentile(d, 95)) if d is not None else float("nan")
    out["assd_mm"] = float(d.mean()) if d is not None else float("nan")
    return out


def structural_agreement(real_masks, gen_masks, spacing):
    """Compare two independently produced segmentations, group by group."""
    results = {}
    for group in CLASS_GROUPS:
        r = _overlap(real_masks[group], gen_masks[group], (spacing,) * 3)
        r["low_resolution_caveat"] = group in LOW_RESOLUTION_CAVEAT
        results[group] = r
    return results


def per_class_intensity(real_CT, generated_CT, real_masks):
    """Absolute / RMS / signed HU error inside reference anatomy.

    Both volumes are expected in the normalized [0, 1] range. The masks must come
    from the *reference* volume: reusing them for the generated volume is exactly
    what makes this an intensity measure rather than a structural one.

    ``bias_hu`` is the informative column. A systematically negative value on
    ``bone`` means the model under-predicts bone density -- a symmetric MAE
    cannot show that, and neither can Dice.
    """
    real_hu = to_hu(np.asarray(real_CT, dtype=np.float64).squeeze())
    gen_hu = to_hu(np.asarray(generated_CT, dtype=np.float64).squeeze())

    results = {}
    for group, mask in real_masks.items():
        if not mask.any():
            results[group] = {"mae_hu": float("nan"), "rmse_hu": float("nan"),
                              "bias_hu": float("nan"), "n_voxels": 0}
            continue
        diff = gen_hu[mask] - real_hu[mask]
        results[group] = {
            "mae_hu": float(np.abs(diff).mean()),
            "rmse_hu": float(np.sqrt((diff ** 2).mean())),
            "bias_hu": float(diff.mean()),
            "n_voxels": int(mask.sum()),
        }
    return results


def evaluate_pair(real_CT, generated_CT, device=None, fast=True):
    """Segment both volumes once and return a flat dict of every group metric.

    Keys are ``<group>_<metric>``. ``dice`` / ``iou`` / ``hd95_mm`` / ``assd_mm``
    come from independent segmentations; ``mae_hu`` / ``bias_hu`` from the
    reference mask.
    """
    real = np.squeeze(np.asarray(
        real_CT.cpu() if isinstance(real_CT, torch.Tensor) else real_CT))
    gen = np.squeeze(np.asarray(
        generated_CT.cpu() if isinstance(generated_CT, torch.Tensor) else generated_CT))
    spacing = spacing_for(real.shape)

    real_masks = segment(real, spacing, device=device, fast=fast)
    gen_masks = segment(gen, spacing, device=device, fast=fast)

    flat = {}
    for group, r in structural_agreement(real_masks, gen_masks, spacing).items():
        for k, v in r.items():
            flat[f"{group}_{k}"] = v
    for group, r in per_class_intensity(real, gen, real_masks).items():
        for k, v in r.items():
            flat[f"{group}_{k}"] = v
    return flat


# ---------------------------------------------------------------------------
# Resolution reference (often mislabelled a "ceiling")
# ---------------------------------------------------------------------------
def segment_ceiling(real_CT_full, target_size=128, device=None, fast=True):
    """How much segmentation agreement the coarse grid alone costs.

    Segments the real CT natively (256^3) and again after downsampling to
    ``target_size``, and compares. Nothing generative is involved. Report it
    beside the generated-volume rows: without it a reader cannot tell a bone Dice
    of 0.82 that is near the practical limit from one that is poor.

    **It is a scale, NOT an upper bound, and it can legitimately be exceeded.**
    The two numbers are not scored the same way:

        here             seg(real@256^3) resampled to 128^3  vs  seg(real@128^3)
        evaluate_pair    seg(gen@128^3)                      vs  seg(real@128^3)

    Both are scored against the same real@128^3 masks, but only this function pays
    a cross-resolution resampling penalty; a reconstruction never does. So a
    perfect reconstruction scores 1.0, not this value, and a good one passes it.
    Do not write "no reconstruction can beat this" -- it is false, and a reader
    who checks it against your own table will find the contradiction.
    """
    full = np.squeeze(np.asarray(
        real_CT_full.cpu() if isinstance(real_CT_full, torch.Tensor) else real_CT_full))
    low = resize_volume(full, target_size)

    masks_full = segment(full, spacing_for(full.shape), device=device, fast=fast)
    masks_low = segment(low, spacing_for(low.shape), device=device, fast=fast)

    # Bring the full-resolution masks onto the low-resolution grid, so the
    # comparison is made where the reconstruction actually lives.
    ref = {}
    for group, m in masks_full.items():
        r = F.interpolate(torch.from_numpy(m).float()[None, None],
                          size=(target_size,) * 3, mode="trilinear",
                          align_corners=False)[0, 0]
        ref[group] = (r > 0.5).numpy()

    return structural_agreement(ref, masks_low, spacing_for(low.shape))
