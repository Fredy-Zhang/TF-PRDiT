"""Segment the real test-set CTs once and cache the label maps.

This is the left-hand side of every comparison, so it is computed once and
reused. Caching it also buys a baseline that matters for interpretation: if
TotalSegmentator finds a structure in every *real* volume, then a structure it
misses in a *reconstruction* is attributable to the reconstruction, not to the
segmenter having a bad day on this dataset. Print the detection counts and check
they are complete before trusting anything downstream.

Writes ``<out-dir>/<vid>.npz`` (key ``labels``, uint8) plus ``volumes.csv`` with
each group's volume in mL -- a quick physiological sanity check: adult lungs are
~3-6 L, and a number far outside that means the spacing or the axis order is
wrong, not that the model is bad.

Usage:
    python scripts/segment_real_testset.py --size 128 --out-dir results/seg_real_128
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_data import load_real, test_ids  # noqa: E402
from utils.seg_metrics import CLASS_GROUPS, group_masks, segment_labels, spacing_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128,
                    help="grid to segment on; match the reconstruction grid")
    ap.add_argument("--data-root", default=None, help="default: data/LIDC-HDF5-256")
    ap.add_argument("--test-txt", default=None, help="default: data_list/test.txt")
    ap.add_argument("--out-dir", default="results/seg_real_128")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    vids = test_ids(args.test_txt)
    if args.limit:
        vids = vids[:args.limit]
    os.makedirs(args.out_dir, exist_ok=True)

    spacing = spacing_for((args.size,) * 3)
    print(f"{len(vids)} volumes, grid {args.size}^3 @ {spacing:.2f} mm -> {args.out_dir}\n")

    rows, found, t0 = [], {g: 0 for g in CLASS_GROUPS}, time.time()
    for i, vid in enumerate(vids, 1):
        npz = os.path.join(args.out_dir, f"{vid}.npz")
        if os.path.exists(npz):
            seg = np.load(npz)["labels"]
            fresh = False
        else:
            vol = load_real(vid, args.size, args.data_root)
            seg = segment_labels(vol, spacing, device=args.device, fast=True)
            np.savez_compressed(npz, labels=seg)
            fresh = True

        masks = group_masks(seg)
        row = {"volume": vid}
        for g, m in masks.items():
            ml = float(m.sum()) * spacing ** 3 / 1000.0
            row[f"{g}_ml"] = ml
            found[g] += bool(m.any())
        rows.append(row)

        print(f"[{i}/{len(vids)}] {vid[:28]:<28} "
              + "  ".join(f"{g}={row[f'{g}_ml']:.0f}mL" for g in CLASS_GROUPS)
              + ("" if fresh else "   (cached)"))

    with open(os.path.join(args.out_dir, "volumes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\ndetection on the real CT -- anything below {len(vids)}/{len(vids)} is a "
          f"segmenter limitation, not a model one:")
    for g in CLASS_GROUPS:
        vals = np.array([r[f"{g}_ml"] for r in rows])
        print(f"  {g:<15} {found[g]:>3}/{len(vids)}   {vals.mean():7.0f} ± "
              f"{vals.std():.0f} mL")
    print(f"\nwrote {args.out_dir}   [{(time.time()-t0)/60:.1f} min]")


if __name__ == "__main__":
    main()
