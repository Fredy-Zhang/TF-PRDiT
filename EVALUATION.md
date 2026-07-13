# Evaluation

MSE / PSNR / SSIM / SNR (already in `util.py`) measure how close the voxels are.
They do not measure whether the reconstruction is *clinically* faithful: a chest
CT is ~30 % air and ~29 % homogeneous soft tissue, so a global error is dominated
by voxels nobody reads, and it cannot say whether the bone is 100 HU too dark or
whether the trachea ended up in the wrong place.

This suite adds two things that can:

| | what it asks | metrics |
|---|---|---|
| **Structural / task-based** | does an off-the-shelf clinical tool recover the same anatomy from the reconstruction? | **Dice**, **IoU**, HD95, ASSD, signed volume error, detection rate |
| **Intensity fidelity** | is the density right *inside* each structure? | **MAE**, RMSE, signed **bias** — all in HU |

The structural metrics segment the reconstruction and the real CT **independently**
with [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) and compare
the two label maps. That independence is the point: per-structure MAE alone invites
the reply *"this is still MSE, just masked"*, because it reuses the reference mask
on both volumes and so cannot notice that a structure has moved.

Six structure groups are scored (`utils/seg_metrics.py: CLASS_GROUPS`): `lung`,
`airway`, `heart`, `great_vessels`, `bone`, `esophagus`. Ribs and vertebrae are
pooled into `bone` — at 2.5 mm a per-rib Dice reflects labelling jitter, not
reconstruction quality.

## Install

```bash
pip install TotalSegmentator            # brings in nnU-Net + torch
```

Segmentation needs a GPU. On a compute node with no egress, download the weights
once and point `TOTALSEG_HOME_DIR` at them:

```bash
export TOTALSEG_HOME_DIR=$HOME/.totalsegmentator
```

## Run it

```bash
# 0. Check the setup BEFORE trusting any number (see "Why this matters" below)
python scripts/validate_eval_setup.py                     # metrics, no GPU
python scripts/validate_eval_setup.py --with-segmentation # + anatomy, needs a GPU

# 1. Segment the real test CTs once; everything else reuses this cache
python scripts/segment_real_testset.py --out-dir results/seg_real_128

# 2. Evaluate one run, or several at once (this is how you get a view sweep).
#    The run dirs are whatever you passed to sample_xrays.py --output-dir.
python scripts/evaluate_segmentation.py --run 2view=outputs_2views
python scripts/evaluate_segmentation.py \
    --run 1view=outputs_1view --run 2view=outputs_2views \
    --run 4view=outputs_4views --run 8view=outputs_8views \
    --out results/views.csv

# 3. Tables
python scripts/summarize_evaluation.py --csv results/views.csv

# 4. Optional: what the 128^3 grid alone costs the segmenter (see below)
python scripts/eval_ceiling.py --out results/ceiling.csv
```

`--run NAME=PATH` takes either the timestamped directory `sample_xrays.py` wrote
or the output directory above it (the newest timestamp is used). Label maps are
cached in `results/seg_<NAME>_128/`, so re-running the tables needs no GPU.

## Reading the results

**Never read Dice against 1.0.** Downsampling the CT to the 128³ reconstruction
grid already costs the segmenter a lot of overlap, and no method is charged for
that separately. `scripts/eval_ceiling.py` measures it by segmenting the *real* CT
at 256³ and again at 128³ (n=102 on LIDC):

| structure | Dice | | structure | Dice |
|---|---|---|---|---|
| lung | 0.976 | | airway | 0.893 |
| heart | 0.960 | | esophagus | 0.855 |
| great_vessels | 0.928 | | **bone** | **0.818** |

Bone tops out around 0.82 purely because of the grid. Without that row, a bone
Dice of 0.75 reads as failure; with it, it is close to what the resolution allows.

**But this is a scale, not a ceiling — it can be exceeded, and that is not a bug.**
The two numbers are not scored the same way:

| | prediction | reference |
|---|---|---|
| `eval_ceiling.py` | `seg(real@256³)` resampled to 128³ | `seg(real@128³)` |
| `evaluate_segmentation.py` | `seg(gen@128³)` | `seg(real@128³)` |

Both are scored against the same real@128³ masks, but only the reference pays a
cross-resolution resampling penalty; a reconstruction never does. **A perfect
reconstruction scores 1.0, not 0.818.** Do not write "no reconstruction can beat
this" — it is false, and a reader who checks it against your own table will find
the contradiction.

**`detected` is not decoration.** A structure the segmenter never finds has no
surface, so its HD95/ASSD are NaN — and a plain mean would silently drop exactly
the model's worst cases, flattering it. Every mean is printed with a detection
rate; if it is not N/N, no mean in that column is unconditional. Note that
`detected=True` with `dice=0.000` is possible and real: the structure was
segmented, just nowhere near the reference.

**Report `bias_hu` next to `mae_hu`.** MAE is symmetric and hides direction. Bone
bias is systematically *negative* (the model under-predicts bone density) and
airway bias strongly *positive* at low view counts (the trachea gets filled in
with soft tissue). MAE shows neither.

**MAE in stored units is already in HU.** Stored value = HU + 1000, and MAE is a
difference, so the offset cancels. The conversion still matters for anything
signed or thresholded (`bias_hu`, the HU bands) — `utils/metrics.py: to_hu`.

## Why the validation script matters

Every metric here rests on assumptions that are *invisible in the output when they
are wrong*: a transposed axis, a stale voxel spacing, or a double-applied HU offset
all yield plausible-looking numbers. A flipped volume still segments happily — it
just puts the anatomy in the wrong place.

`scripts/validate_eval_setup.py` asserts them, and it is worth re-running after any
change to `utils/seg_metrics.py`:

- **Spacing.** The `spacing` field in the h5 files says 1.0 mm and is **stale**.
  The volumes are 256³ resamplings of a 320 mm field of view, so true spacing is
  **1.25 mm at 256³ → 2.50 mm at 128³**. Confirmed against body extent: 1.25 mm
  gives a plausible ~250 mm AP thorax, 1.0 mm an implausible ~200 mm. Get this
  wrong and every mm-valued metric is off by 25 %.
- **Axis order** is `(z, y, x)`, +x to the patient's left, +y posterior, +z
  inferior — a left-handed frame, so the NIfTI affine has a negative determinant.
  That is valid, and nnU-Net reorients from it. Four anatomical assertions check
  it (upper lobes superior to lower, left lung lateral, vertebrae dorsal to the
  heart, physiological lung volume).
- **Metric behaviour** on synthetic masks with known answers (identical → Dice 1 /
  ASSD 0; half-overlap → Dice 0.5, IoU 1/3; one-voxel dilation → HD95 exactly one
  voxel; disjoint → Dice 0 but `detected=True`; missed → NaN distances).

## Caveats worth stating in a paper

- **HD95 saturates at one voxel (2.50 mm) on the 128³ grid.** Once two surfaces
  are within a voxel — which is where good reconstructions sit — it stops
  discriminating. `assd_mm` averages instead of thresholding and still separates
  them. Keep HD95 anyway: it regains its bite on poor reconstructions.
- `airway`, `esophagus` and `great_vessels` are thin or small at 2.5 mm and are
  flagged `*` in the tables. A low score there is partly the grid, not the model.
- **RAD-ChestCT is normalized differently** (`datasets/rad_chest.py` applies a
  *per-volume* 0.995-quantile clip), so a normalized volume cannot be mapped back
  to HU with one global constant. The HU numbers above are LIDC-specific; for
  RAD-ChestCT pass the right `ct_min`/`ct_max` or read them as normalized units.
