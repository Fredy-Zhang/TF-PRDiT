"""Measure what the reconstruction grid alone costs the segmenter.

Segments each real CT twice -- once natively at 256^3, once after downsampling to
the reconstruction grid (128^3) -- and reports the agreement. No generative model
is involved. Report it beside the generated-volume rows: without it, a reader
cannot tell a bone Dice of 0.82 that is near the practical limit of this grid from
one that is poor.

**This is a scale, not a ceiling.** It is tempting to call it an upper bound that
"no reconstruction can beat", and that is false. The two numbers are not scored
the same way:

    here                      seg(real@256^3) resampled to 128^3  vs  seg(real@128^3)
    evaluate_segmentation.py  seg(gen@128^3)                      vs  seg(real@128^3)

Both are scored against the same real@128^3 masks, but only this script pays a
cross-resolution resampling penalty; a reconstruction never does. A reconstruction
that reproduces the 128^3 volume exactly therefore scores 1.0, not this value, and
a good one exceeds it. Read it as "a Dice difference of this size is what halving
the grid costs" -- never as an achievable maximum.

Usage (needs a GPU):
    python scripts/eval_ceiling.py --limit 102 --out results/ceiling.csv
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_data import load_real, test_ids  # noqa: E402
from utils.seg_metrics import CLASS_GROUPS, segment_ceiling  # noqa: E402

METRICS = ["dice", "iou", "hd95_mm", "assd_mm", "rel_vol_err",
           "vol_ref_ml", "vol_pred_ml", "detected"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-size", type=int, default=128,
                    help="the reconstruction grid")
    ap.add_argument("--limit", type=int, default=None, help="default: whole test split")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--test-txt", default=None)
    ap.add_argument("--out", default="results/ceiling.csv")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    vids = test_ids(args.test_txt)
    if args.limit:
        vids = vids[:args.limit]
    print(f"{len(vids)} volume(s): TotalSegmentator(real@256^3) vs "
          f"TotalSegmentator(real@{args.target_size}^3)\n")

    rows, t0 = [], time.time()
    for i, vid in enumerate(vids, 1):
        # Native resolution: no resize, so the h5 is loaded at its stored 256^3.
        full = load_real(vid, size=256, data_root=args.data_root)
        res = segment_ceiling(full, target_size=args.target_size, device=args.device)
        for group, r in res.items():
            rows.append({"volume": vid, "group": group,
                         **{k: r[k] for k in METRICS}})
        print(f"[{i}/{len(vids)}] {vid[:28]:<28} "
              + "  ".join(f"{g}={res[g]['dice']:.3f}" for g in CLASS_GROUPS))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["volume", "group"] + METRICS)
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'structure':<16}{'Dice':>8}{'IoU':>8}{'ASSD mm':>10}{'vol err':>10}")
    for g in CLASS_GROUPS:
        sel = [r for r in rows if r["group"] == g]
        d = np.nanmean([r["dice"] for r in sel])
        i_ = np.nanmean([r["iou"] for r in sel])
        a = np.nanmean([r["assd_mm"] for r in sel])
        v = np.nanmean([r["rel_vol_err"] for r in sel])
        print(f"{g:<16}{d:>8.3f}{i_:>8.3f}{a:>10.2f}{100*v:>9.1f}%")
    print(f"\nA scale for reading generated-volume Dice -- NOT an upper bound; "
          f"a good reconstruction exceeds it.")
    print(f"wrote {args.out}   [{(time.time()-t0)/60:.1f} min]")


if __name__ == "__main__":
    main()
