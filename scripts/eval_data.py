"""Locating and loading the volumes the evaluation scripts compare.

One module so that "what is the reference" and "what is the reconstruction" are
answered in exactly one place. Every loader returns a **normalized [0, 1] float32
array in (z, y, x) order**, which is the convention the rest of the evaluation
assumes.

A *run* is one sampling output directory, i.e. what ``sample_xrays.py`` writes:

    <output_dir>/<timestamp>/samples/<sample_id>/generated/sample_0_0.nii.gz
                                                /reference/...

Pass either the ``<timestamp>`` directory or the ``<output_dir>`` above it (the
newest timestamp is then used). Name the runs to compare view counts:

    --run 1view=outputs_xrays_1 --run 2view=outputs_xrays_2 ...

**Do not use the volume in ``reference/`` as ground truth.** It is written in the
diffusion's own space, not in [0, 1]. The reference always comes from the h5 via
``load_real``, which is also what the cached reference segmentation was built
from -- so the labels and the intensities always describe the same volume.
"""
import glob
import os

import h5py
import nibabel as nib
import numpy as np

from utils.metrics import CT_MIN, CT_MAX
from utils.seg_metrics import resize_volume

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GENERATED_NAME = "sample_0_0.nii.gz"  # save_evaluation_samples: sample_<i>_<epoch>


def test_ids(test_txt=None):
    """The evaluation split, in file order."""
    test_txt = test_txt or os.path.join(ROOT, "data_list/test.txt")
    with open(test_txt) as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_real(vid, size=128, data_root=None):
    """Reference CT from the h5, resized to ``size``, normalized to [0, 1].

    This is the single definition of "the real volume". Keep it identical
    everywhere: the cached reference segmentation must describe the same voxels
    the intensity metrics are computed on.
    """
    data_root = data_root or os.path.join(ROOT, "data/LIDC-HDF5-256")
    path = os.path.join(data_root, vid, "ct_xray_data.h5")
    with h5py.File(path, "r") as h:
        ct = np.asarray(h["ct"]).astype(np.float32)
    ct = (np.clip(ct, CT_MIN, CT_MAX) - CT_MIN) / (CT_MAX - CT_MIN)
    if size != ct.shape[-1]:
        ct = resize_volume(ct, size).numpy()
    return ct


def resolve_run(path):
    """Accept a timestamp dir or the output dir above it; return the one with samples/."""
    if os.path.isdir(os.path.join(path, "samples")):
        return path
    stamps = sorted(d for d in glob.glob(os.path.join(path, "*"))
                    if os.path.isdir(os.path.join(d, "samples")))
    if not stamps:
        raise FileNotFoundError(
            f"{path!r} contains no samples/ directory, and no timestamped run below it "
            f"does either. Point --run at what sample_xrays.py wrote.")
    return stamps[-1]  # newest timestamp


def parse_runs(specs):
    """``["1view=outputs/a", "outputs/b"]`` -> ``{"1view": <dir>, "b": <dir>}``."""
    runs = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        if not path:
            name, path = os.path.basename(os.path.normpath(name)), name
        runs[name] = resolve_run(path)
    return runs


def generated_path(run_dir, vid):
    return os.path.join(run_dir, "samples", vid, "generated", GENERATED_NAME)


def load_generated(run_dir, vid):
    """Reconstruction for one case, normalized to [0, 1], in (z, y, x) order.

    ``sample_xrays.py`` already clamps to [0, 1] before saving, so this only
    squeezes and re-clips. No transpose and no flip: the saved volume is in the
    same (z, y, x) frame as the h5 reference.
    """
    vol = np.asarray(nib.load(generated_path(run_dir, vid)).dataobj).astype(np.float32)
    vol = np.squeeze(vol)
    if vol.ndim != 3:
        raise ValueError(f"{vid}: expected a 3D volume, got shape {vol.shape}")
    return np.clip(vol, 0.0, 1.0)


def covered_ids(run_dir, vids):
    """Which of ``vids`` this run actually produced a volume for."""
    return [v for v in vids if os.path.exists(generated_path(run_dir, v))]
