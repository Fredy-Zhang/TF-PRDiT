"""Self-check the evaluation setup. Run this before trusting any number it produces.

Every metric here depends on assumptions that are invisible in the output if they
are wrong: a transposed axis, a stale voxel spacing, or an HU offset applied twice
all produce plausible-looking numbers. This script asserts them.

Part 1 (no GPU, no data) -- the metric implementations, on synthetic masks with
known answers, and the HU conversion.

Part 2 (--with-segmentation; needs a GPU and the data) -- runs TotalSegmentator on
one real volume and asserts four *anatomical* facts. These are the ones that catch
a wrong axis order or a wrong affine, because a flipped volume still segments
happily; it just puts the structures in the wrong place:
    upper lobes superior to lower lobes · left lung on the patient's left ·
    vertebrae dorsal to the heart · lung volume physiologically plausible

Usage:
    python scripts/validate_eval_setup.py
    python scripts/validate_eval_setup.py --with-segmentation
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy import ndimage  # noqa: E402

from utils.metrics import to_hu  # noqa: E402
from utils.seg_metrics import _overlap, spacing_for  # noqa: E402

OK, FAIL = "  ok  ", " FAIL "
_failures = []


def check(name, cond, detail=""):
    print(f"[{OK if cond else FAIL}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        _failures.append(name)


def check_close(name, got, want, tol, unit=""):
    cond = abs(got - want) <= tol
    check(name, cond, f"got {got:.4f}{unit}, expected {want:.4f}{unit} (±{tol}{unit})")


def part1():
    print("== spacing ==")
    # The h5 `spacing` field says 1.0 mm and is stale: these are 256^3 resamplings
    # of a 320 mm FOV. Get this wrong and every mm-valued metric is wrong by 25%.
    check_close("256^3 -> 1.25 mm", spacing_for((256,) * 3), 1.25, 1e-9, " mm")
    check_close("128^3 -> 2.50 mm", spacing_for((128,) * 3), 2.50, 1e-9, " mm")

    print("\n== HU conversion ==")
    # Stored value = HU + 1000. The offset cancels in a difference, which is why an
    # MAE computed in stored units is already in HU -- assert it, don't trust it.
    check_close("normalized 0.4 -> HU", float(to_hu(np.array(0.4))), 0.0, 1e-9, " HU")
    check_close("normalized 0.0 -> HU", float(to_hu(np.array(0.0))), -1000.0, 1e-9, " HU")
    a, b = np.array([0.4, 0.5, 0.6]), np.array([0.44, 0.5, 0.56])
    mae_stored = float(np.abs(a - b).mean() * 2500)
    mae_hu = float(np.abs(to_hu(a) - to_hu(b)).mean())
    check_close("MAE in stored units == MAE in HU", mae_stored, mae_hu, 1e-6, " HU")

    print("\n== overlap metrics, synthetic masks (2.50 mm voxels) ==")
    sp = (2.5, 2.5, 2.5)
    ref = np.zeros((40, 40, 40), dtype=bool)
    ref[12:28, 12:28, 12:28] = True

    r = _overlap(ref, ref.copy(), sp)
    check_close("identical: Dice", r["dice"], 1.0, 1e-9)
    check_close("identical: IoU", r["iou"], 1.0, 1e-9)
    check_close("identical: ASSD", r["assd_mm"], 0.0, 1e-9, " mm")
    check_close("identical: HD95", r["hd95_mm"], 0.0, 1e-9, " mm")
    check_close("identical: vol err", r["rel_vol_err"], 0.0, 1e-9)
    check("identical: detected", r["detected"] is True)

    # A known Dice/IoU pair: two boxes overlapping in exactly half their volume.
    half = np.zeros_like(ref)
    half[12:28, 12:28, 20:36] = True          # shifted 8 voxels along the last axis
    r = _overlap(ref, half, sp)
    check_close("half-overlap: Dice", r["dice"], 0.5, 1e-9)
    check_close("half-overlap: IoU", r["iou"], 1.0 / 3.0, 1e-9)  # Dice/(2-Dice)

    # One voxel of dilation must read as exactly one voxel of surface distance.
    grown = ndimage.binary_dilation(ref)
    r = _overlap(ref, grown, sp)
    check_close("1-voxel dilation: HD95", r["hd95_mm"], 2.5, 1e-6, " mm")
    check("1-voxel dilation: volume grows", r["rel_vol_err"] > 0,
          f"rel_vol_err = {r['rel_vol_err']:+.1%}")

    # Disjoint but non-empty: Dice 0 with detected=True. This distinction is the
    # whole point of `detected` -- the structure was segmented, just in the wrong
    # place, and that is NOT the same as the segmenter missing it.
    far = np.zeros_like(ref)
    far[30:38, 30:38, 30:38] = True
    r = _overlap(ref, far, sp)
    check_close("disjoint: Dice", r["dice"], 0.0, 1e-9)
    check("disjoint: detected is still True", r["detected"] is True)
    check("disjoint: ASSD is finite", np.isfinite(r["assd_mm"]),
          f"assd = {r['assd_mm']:.1f} mm")

    # A missed structure: distances are NaN, and `detected` is what stops a
    # downstream nanmean from quietly dropping the model's worst failure.
    r = _overlap(ref, np.zeros_like(ref), sp)
    check("missed: detected is False", r["detected"] is False)
    check("missed: ASSD is NaN", np.isnan(r["assd_mm"]))
    check_close("missed: Dice", r["dice"], 0.0, 1e-9)


def part2(args):
    import torch  # noqa: F401
    from scripts.eval_data import load_real, test_ids
    from utils.seg_metrics import CLASS_GROUPS, group_masks, segment_labels

    from totalsegmentator.map_to_binary import class_map
    name_to_idx = {v: k for k, v in class_map["total"].items()}

    vid = args.volume or test_ids(args.test_txt)[0]
    size = args.size
    print(f"\n== anatomy, {vid} at {size}^3 ==")
    vol = load_real(vid, size, args.data_root)
    spacing = spacing_for(vol.shape)
    seg = segment_labels(vol, spacing, device=args.device, fast=True)
    masks = group_masks(seg)

    def com(label):
        idx = name_to_idx.get(label)
        m = seg == idx
        return ndimage.center_of_mass(m) if m.any() else None

    # Axis order is (z, y, x): +z inferior, +y posterior, +x to the patient's left.
    upper = com("lung_upper_lobe_left")
    lower = com("lung_lower_lobe_left")
    check("upper lobe is superior to lower lobe (smaller z)",
          upper and lower and upper[0] < lower[0],
          f"z: upper {upper[0]:.1f} < lower {lower[0]:.1f}" if upper and lower else "missing")

    left = com("lung_upper_lobe_left")
    right = com("lung_upper_lobe_right")
    check("left lung is on the patient's left (larger x)",
          left and right and left[2] > right[2],
          f"x: left {left[2]:.1f} > right {right[2]:.1f}" if left and right else "missing")

    heart = com("heart")
    vert = com("vertebrae_T8")
    check("vertebrae are dorsal to the heart (larger y)",
          heart and vert and vert[1] > heart[1],
          f"y: vertebra {vert[1]:.1f} > heart {heart[1]:.1f}" if heart and vert else "missing")

    lung_ml = float(masks["lung"].sum()) * spacing ** 3 / 1000.0
    check("lung volume is physiological (2-8 L)", 2000 < lung_ml < 8000,
          f"{lung_ml:.0f} mL")

    print("\n  (a wrong axis order or affine still segments happily -- it just puts")
    print("   the anatomy in the wrong place, which is exactly what these catch.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-segmentation", action="store_true",
                    help="also run TotalSegmentator and assert anatomy (needs a GPU)")
    ap.add_argument("--volume", default=None)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--test-txt", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    part1()
    if args.with_segmentation:
        part2(args)
    else:
        print("\n(skipping the anatomy checks; pass --with-segmentation to run them)")

    print()
    if _failures:
        sys.exit(f"{len(_failures)} check(s) FAILED: {', '.join(_failures)}")
    print("all checks passed")


if __name__ == "__main__":
    main()
