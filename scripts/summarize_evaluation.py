"""Aggregate the per-case evaluation into the tables that go in the paper.

Reads a ``scripts/evaluate_segmentation.py`` CSV and prints, per structure group
and per run: detection rate, Dice, IoU, ASSD, HD95, signed volume error, and the
HU errors (MAE / RMSE / bias). Global MSE / PSNR / SSIM are printed alongside, so
the segmentation metrics are read *next to* the familiar ones rather than instead
of them.

Two rules are enforced here rather than left to the reader:

  * **Detection rate is printed above every mean.** A structure the segmenter
    never finds has no surface distance; those entries are NaN, and a plain mean
    would silently drop them -- which flatters the model by discarding exactly its
    worst cases. If a detection rate is not N/N, no mean in its column is
    unconditional.
  * **The resolution reference is printed with Dice, if available.** Dice at
    2.5 mm is not interpretable against 1.0; see ``eval_ceiling.py``. It is a
    scale, not an upper bound, and it can be exceeded.

Usage:
    python scripts/summarize_evaluation.py --csv results/evaluation.csv
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seg_metrics import CLASS_GROUPS, LOW_RESOLUTION_CAVEAT  # noqa: E402

GROUPS = list(CLASS_GROUPS)


def load(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k in ("volume", "run"):
                continue
            if v in ("True", "False"):
                r[k] = v == "True"
            else:
                r[k] = float(v) if v not in ("", "nan", "None") else float("nan")
    return rows


def agg(vals):
    v = np.asarray(vals, dtype=np.float64)
    ok = v[~np.isnan(v)]
    if not ok.size:
        return float("nan"), float("nan")
    return ok.mean(), ok.std()


def reference_dice(path):
    """Per-group Dice from eval_ceiling.py, if it has been run."""
    if not path or not os.path.exists(path):
        return {}, 0
    per, vols = {}, set()
    with open(path) as f:
        for r in csv.DictReader(f):
            per.setdefault(r["group"], []).append(float(r["dice"]))
            vols.add(r["volume"])
    return {g: float(np.mean(v)) for g, v in per.items()}, len(vols)


def table(rows, runs, metric, fmt, scale=1.0, title="", note=""):
    print(f"\n### {title or metric}{'  ' + note if note else ''}")
    print(f"{'structure':<16}" + "".join(f"{n:>20}" for n in runs))
    fmt_sd = fmt.replace("+", "")  # a standard deviation is never signed
    for g in GROUPS:
        cells = []
        for n in runs:
            sel = [r for r in rows if r["run"] == n]
            mu, sd = agg([r[f"{g}_{metric}"] * scale for r in sel])
            cells.append(f"{fmt % mu} ± {fmt_sd % sd}")
        star = " *" if g in LOW_RESOLUTION_CAVEAT else ""
        print(f"{g + star:<16}" + "".join(f"{c:>20}" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/evaluation.csv")
    ap.add_argument("--reference", default="results/ceiling.csv",
                    help="eval_ceiling.py output; printed beside Dice if present")
    ap.add_argument("--out", default="results/evaluation_summary.csv")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    rows = load(args.csv)
    runs = list(dict.fromkeys(r["run"] for r in rows))
    vids = sorted({r["volume"] for r in rows})
    print(f"{len(vids)} volumes x {len(runs)} run(s) from {args.csv}")

    ref, n_ref = reference_dice(args.reference)

    print(f"\n### global image metrics (the familiar ones, for context)")
    print(f"{'run':<16}{'MSE':>12}{'PSNR dB':>10}{'SSIM':>9}"
          f"{'MAE HU':>10}{'RMSE HU':>10}")
    for n in runs:
        sel = [r for r in rows if r["run"] == n]
        cells = [agg([r[k] for r in sel])[0]
                 for k in ("mse", "psnr", "ssim", "mae_global_hu", "rmse_global_hu")]
        print(f"{n:<16}{cells[0]:>12.5f}{cells[1]:>10.2f}{cells[2]:>9.4f}"
              f"{cells[3]:>10.1f}{cells[4]:>10.1f}")

    print(f"\n### MAE by HU tissue band (HU) -- where the global MAE actually comes from")
    bands = ["air", "lung_parenchyma", "fat", "soft_tissue", "bone"]
    print(f"{'run':<16}" + "".join(f"{b[:12]:>15}" for b in bands))
    for n in runs:
        sel = [r for r in rows if r["run"] == n]
        cells = [f"{agg([r[f'mae_{b}'] for r in sel])[0]:.0f}" for b in bands]
        print(f"{n:<16}" + "".join(f"{c:>15}" for c in cells))
    frac = [agg([r[f"frac_{b}"] for r in rows])[0] for b in bands]
    print(f"{'(voxel share)':<16}" + "".join(f"{100*f:>14.1f}%" for f in frac))

    # Detection first: every mean below is conditional on the structure existing.
    print(f"\n### detection rate (structure found in the reconstruction)")
    print(f"{'structure':<16}" + "".join(f"{n:>20}" for n in runs))
    for g in GROUPS:
        cells = []
        for n in runs:
            sel = [r for r in rows if r["run"] == n]
            k = sum(bool(r[f"{g}_detected"]) for r in sel)
            cells.append(f"{k}/{len(sel)}  ({100*k/len(sel):.0f}%)")
        print(f"{g:<16}" + "".join(f"{c:>20}" for c in cells))

    table(rows, runs, "dice", "%.3f", title="Dice (higher better)")
    table(rows, runs, "iou", "%.3f", title="IoU (higher better)")
    table(rows, runs, "assd_mm", "%.2f", title="ASSD, mm (lower better)")
    table(rows, runs, "hd95_mm", "%.2f", title="HD95, mm (lower better)",
          note="-- saturates at one voxel (2.50 mm at 128^3)")
    table(rows, runs, "rel_vol_err", "%+.1f", scale=100.0,
          title="relative volume error, % (signed)")
    table(rows, runs, "mae_hu", "%.1f", title="MAE, HU (inside reference anatomy)")
    table(rows, runs, "rmse_hu", "%.1f", title="RMSE, HU")
    table(rows, runs, "bias_hu", "%+.1f", title="bias, HU (signed)",
          note="-- negative = under-predicts density")

    if ref:
        print(f"\n### Dice vs the resolution reference  (real CT 256^3 vs 128^3, n={n_ref})")
        print("    What the coarse grid alone costs the segmenter. A SCALE, not a bound:")
        print("    the reference is scored across two grids and the runs within one, so a")
        print("    good reconstruction can exceed it. A perfect one would score 1.000.")
        print(f"{'structure':<16}{'reference':>11}" + "".join(f"{n:>12}" for n in runs))
        for g in GROUPS:
            if g not in ref:
                continue
            cells = [f"{agg([r[f'{g}_dice'] for r in rows if r['run'] == n])[0]:.3f}"
                     for n in runs]
            print(f"{g:<16}{ref[g]:>11.3f}" + "".join(f"{c:>12}" for c in cells))

    fields = ["run", "structure", "n", "detected", "dice_mean", "dice_std",
              "iou_mean", "iou_std", "assd_mean", "assd_std", "hd95_mean", "hd95_std",
              "rel_vol_err_mean", "rel_vol_err_std", "mae_hu_mean", "mae_hu_std",
              "rmse_hu_mean", "rmse_hu_std", "bias_hu_mean", "bias_hu_std"]
    out = []
    for n in runs:
        sel = [r for r in rows if r["run"] == n]
        for g in GROUPS:
            rec = {"run": n, "structure": g, "n": len(sel),
                   "detected": sum(bool(r[f"{g}_detected"]) for r in sel)}
            for metric, key in [("dice", "dice"), ("iou", "iou"), ("assd_mm", "assd"),
                                ("hd95_mm", "hd95"), ("rel_vol_err", "rel_vol_err"),
                                ("mae_hu", "mae_hu"), ("rmse_hu", "rmse_hu"),
                                ("bias_hu", "bias_hu")]:
                mu, sd = agg([r[f"{g}_{metric}"] for r in sel])
                rec[f"{key}_mean"], rec[f"{key}_std"] = mu, sd
            out.append(rec)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {args.out}")
    print("* thin/small at 2.5 mm -- a low score here is partly the grid, not the model")


if __name__ == "__main__":
    main()
