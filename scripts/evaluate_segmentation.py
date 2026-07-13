"""Evaluate reconstructions against the real CT: TotalSegmentator + MAE/IoU.

For every case in every run this computes three families of metric, and they are
deliberately kept apart because they answer different questions:

  1. **Structural / task-based** -- the reconstruction and the real CT are
     segmented *independently* with TotalSegmentator, and the two label maps are
     compared: Dice, **IoU**, HD95, ASSD, signed volume error, detection.
     This is the one that can say a structure is in the wrong place, and it is
     what answers "do MSE/PSNR/SSIM capture clinical quality" with evidence.
  2. **Intensity fidelity** -- **MAE**, RMSE and signed bias in HU inside anatomy
     delineated on the *reference* volume (so both volumes see the same voxels).
     This says how right the density is, not where the structure is.
  3. **Global image metrics** -- MSE / PSNR / SSIM from ``util.py``, plus MAE and
     RMSE stratified by HU tissue band. Reported so the new metrics can be read
     next to the familiar ones, not instead of them.

Label maps are cached per run in ``<cache-root>/seg_<run>_<size>/<vid>.npz``, so
re-running is cheap and the tables can be rebuilt with no GPU.

Usage -- one run, or a whole view sweep in one command:
    python scripts/evaluate_segmentation.py --run 2view=outputs_xrays_2
    python scripts/evaluate_segmentation.py \
        --run 1view=outputs_xrays_1 --run 2view=outputs_xrays_2 \
        --out results/views.csv

Needs a GPU. TotalSegmentator's weights must already be present on a node with no
egress: set TOTALSEG_HOME_DIR to the directory holding them.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_data import covered_ids, load_generated, load_real, parse_runs, test_ids  # noqa: E402
from util import evaluation_metrics  # noqa: E402
from utils.metrics import hu_band_metrics  # noqa: E402
from utils.seg_metrics import (  # noqa: E402
    CLASS_GROUPS, group_masks, per_class_intensity, segment_labels, spacing_for,
    structural_agreement,
)

# Structural metrics come from independent segmentations; intensity ones from the
# reference mask. Both are per group, so they share the <group>_<metric> namespace.
STRUCT_KEYS = ["dice", "iou", "hd95_mm", "assd_mm", "rel_vol_err",
               "vol_ref_ml", "vol_pred_ml", "detected"]
INTENSITY_KEYS = ["mae_hu", "rmse_hu", "bias_hu", "n_voxels"]


def labels_for(name, vid, volume_fn, size, cache_root, device):
    """Cached TotalSegmentator label map, segmenting on a cache miss."""
    cache = os.path.join(cache_root, f"seg_{name}_{size}")
    os.makedirs(cache, exist_ok=True)
    npz = os.path.join(cache, f"{vid}.npz")
    if os.path.exists(npz):
        return np.load(npz)["labels"], False

    vol = volume_fn()
    if vol.shape != (size,) * 3:
        raise ValueError(f"{name}/{vid}: expected {size}^3, got {vol.shape}")
    seg = segment_labels(vol, spacing_for(vol.shape), device=device, fast=True)
    np.savez_compressed(npz, labels=seg)
    return seg, True


def global_metrics(real, gen):
    """MSE/PSNR/SSIM (util.py) + HU-band MAE/RMSE. Both volumes in [0, 1]."""
    m = evaluation_metrics(real[None], gen[None])
    out = {k: float(v) for k, v in m.items()
           if k in ("mse", "psnr", "ssim", "snr")}
    out.update(hu_band_metrics(real, gen))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="NAME=PATH",
                    help="a sampling output dir; repeatable, e.g. --run 2view=outputs_xrays_2")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--volume", action="append", default=None,
                    help="restrict to these ids; default = the whole test split")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--test-txt", default=None)
    ap.add_argument("--cache-root", default="results")
    ap.add_argument("--real-cache", default=None,
                    help="default: <cache-root>/seg_real_<size>")
    ap.add_argument("--out", default="results/evaluation.csv")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    runs = parse_runs(args.run)
    real_cache = args.real_cache or os.path.join(args.cache_root, f"seg_real_{args.size}")
    vids = args.volume or test_ids(args.test_txt)
    if args.limit:
        vids = vids[:args.limit]

    spacing = spacing_for((args.size,) * 3)
    print(f"{len(runs)} run(s) x {len(vids)} volume(s), grid {args.size}^3 "
          f"@ {spacing:.2f} mm")
    for name, d in runs.items():
        n = len(covered_ids(d, vids))
        print(f"  {name:<12} {n:>3}/{len(vids)} volumes   {d}")
        if n == 0:
            sys.exit(f"run {name!r} produced none of the requested volumes -- "
                     f"is it still sampling, or was --num-save-samples too small?")
    print()

    rows, t0 = [], time.time()
    for i, vid in enumerate(vids, 1):
        real_vol = load_real(vid, args.size, args.data_root)
        # Prefer an existing reference cache (scripts/segment_real_testset.py, or a
        # shared one) and only segment the real volume if there is none -- checking
        # this *before* segmenting is what makes the reference free on a re-run.
        cached_real = os.path.join(real_cache, f"{vid}.npz")
        if os.path.exists(cached_real):
            real_seg = np.load(cached_real)["labels"]
        else:
            real_seg, _ = labels_for("real", vid, lambda: real_vol, args.size,
                                     args.cache_root, args.device)
        real_masks = group_masks(real_seg)

        for name, run_dir in runs.items():
            try:
                gen_vol = load_generated(run_dir, vid)
                seg, fresh = labels_for(name, vid, lambda: gen_vol, args.size,
                                        args.cache_root, args.device)
            except FileNotFoundError:
                print(f"[{i}/{len(vids)}] {vid[:24]:<24} {name:<10} missing, skipped")
                continue
            except Exception as e:
                print(f"[{i}/{len(vids)}] {vid[:24]:<24} {name:<10} FAILED ({e})")
                continue

            struct = structural_agreement(real_masks, group_masks(seg), spacing)
            inten = per_class_intensity(real_vol, gen_vol, real_masks)

            row = {"volume": vid, "run": name}
            row.update(global_metrics(real_vol, gen_vol))
            for g in CLASS_GROUPS:
                for k in STRUCT_KEYS:
                    row[f"{g}_{k}"] = struct[g][k]
                for k in INTENSITY_KEYS:
                    row[f"{g}_{k}"] = inten[g][k]
            rows.append(row)

            dices = "  ".join(f"{g}={struct[g]['dice']:.3f}" for g in CLASS_GROUPS)
            missed = [g for g in CLASS_GROUPS if not struct[g]["detected"]]
            print(f"[{i}/{len(vids)}] {vid[:24]:<24} {name:<10} {dices}"
                  + (f"   MISSED: {','.join(missed)}" if missed else "")
                  + ("" if fresh else "   (cached)"))

    if not rows:
        sys.exit("no volume was evaluated")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    names = list(runs)
    print(f"\n{'structure':<16}" + "".join(f"{n:>20}" for n in names))
    for g in CLASS_GROUPS:
        cells = []
        for n in names:
            sel = [r for r in rows if r["run"] == n]
            d = np.array([r[f"{g}_dice"] for r in sel], dtype=np.float64)
            det = int(sum(bool(r[f"{g}_detected"]) for r in sel))
            # The detection count sits next to the mean on purpose: a nanmean over
            # a structure the model missed would otherwise hide its worst cases.
            cells.append(f"{np.nanmean(d):.3f} ({det}/{len(sel)})")
        print(f"{g:<16}" + "".join(f"{c:>20}" for c in cells))
    print("\nDice, mean over volumes, (detected/total) in parens.")
    print(f"wrote {args.out}   [{(time.time()-t0)/60:.1f} min]")
    print("Now: python scripts/summarize_evaluation.py --csv " + args.out)


if __name__ == "__main__":
    main()
