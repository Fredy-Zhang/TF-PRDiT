"""Hounsfield-unit conversion and tissue-stratified error metrics.

``util.py`` already carries the global image metrics (MSE / PSNR / SSIM / SNR).
Those summarise a whole volume with one number, and a reviewer's objection to
them is fair: a chest CT is mostly air and homogeneous soft tissue, so a global
MSE is dominated by voxels nobody looks at, and it cannot tell you whether the
*bone* is 100 HU too dark.

This module adds the part that can be interpreted clinically:

  * ``to_hu`` -- map the model's normalized [0, 1] range back to Hounsfield
    units, so every error below is quoted in the unit a radiologist reads.
  * ``hu_band_metrics`` -- MAE / RMSE stratified by tissue band (air, lung
    parenchyma, fat, soft tissue, bone). Zero extra dependencies, so it is always
    computable even when TotalSegmentator is unavailable.

The HU convention, for LIDC-IDRI as preprocessed by the X2CT-GAN pipeline this
repo consumes: **stored voxel value = HU + 1000**, clipped to [0, 2500], i.e.
HU in [-1000, 1500]. Note the useful consequence -- MAE is a *difference*, so the
+1000 offset cancels and an MAE computed in stored units is already in HU. The
conversion still matters for anything signed or thresholded (bias, HU bands).

RAD-ChestCT is preprocessed differently (``datasets/rad_chest.py`` clips to
[-1000, 1000] and then applies a *per-volume* 0.995-quantile clip before
normalising). That upper bound is data dependent, so a normalized RAD-ChestCT
volume cannot be mapped back to HU with a single global constant. Pass the right
``ct_min`` / ``ct_max`` explicitly, or read these metrics as normalized units.
"""
import numpy as np

# LIDC-IDRI (X2CT-GAN preprocessing): stored value = HU + 1000, clipped [0, 2500].
CT_MIN, CT_MAX = 0, 2500
HU_OFFSET = -1000

# Tissue bands in HU. The ranges are deliberately non-contiguous: the gaps are
# transition voxels whose tissue class is genuinely ambiguous at 2.5 mm spacing,
# and forcing them into a neighbouring band would only add noise to that band.
HU_BANDS = {
    "air": (-1024, -900),
    "lung_parenchyma": (-900, -500),
    "fat": (-150, -50),
    "soft_tissue": (-50, 100),
    "bone": (300, 1500),
}


def to_hu(image, ct_min=CT_MIN, ct_max=CT_MAX, hu_offset=HU_OFFSET):
    """Map a volume from the normalized [0, 1] range back to Hounsfield units."""
    return np.asarray(image, dtype=np.float64) * (ct_max - ct_min) + ct_min + hu_offset


def hu_band_metrics(real_CT, generated_CT, bands=HU_BANDS, ct_min=CT_MIN, ct_max=CT_MAX):
    """Stratify absolute error by the tissue band each *reference* voxel falls in.

    Both volumes are expected in the normalized [0, 1] range.

    Bands are defined on the reference volume only. That is deliberate: it fixes
    one voxel set per band, so the same voxels are scored for every model being
    compared and the bands cannot shift under a model that hallucinates bone.

    Returns a flat dict of ``mae_<band>`` / ``rmse_<band>`` in HU, plus
    ``frac_<band>`` -- the share of the volume the band covers, which is what
    stops a spectacular ``mae_bone`` from being read without noticing that bone
    is 2 % of the voxels.
    """
    real_hu = to_hu(real_CT, ct_min, ct_max)
    gen_hu = to_hu(generated_CT, ct_min, ct_max)
    err = np.abs(real_hu - gen_hu)

    out = {}
    n_total = real_hu.size
    for name, (lo, hi) in bands.items():
        mask = (real_hu >= lo) & (real_hu < hi)
        n = int(mask.sum())
        out[f"frac_{name}"] = n / n_total
        if n == 0:
            out[f"mae_{name}"] = float("nan")
            out[f"rmse_{name}"] = float("nan")
            continue
        e = err[mask]
        out[f"mae_{name}"] = float(e.mean())
        out[f"rmse_{name}"] = float(np.sqrt((e ** 2).mean()))

    out["mae_global_hu"] = float(err.mean())
    out["rmse_global_hu"] = float(np.sqrt((err ** 2).mean()))
    return out
