# Scripts

This folder lists the maintained helper scripts needed for the sampling workflow.

## Maintained Entry Points

| Script | Purpose |
|---|---|
| `download_lidc_idri.py` | Download the X2CT-GAN preprocessed LIDC-IDRI HDF5 dataset. |
| `ct2xrays.py` | Generate DRR/X-ray projections from CT volumes with DiffDRR. |

## Evaluation

TotalSegmentator-based structural metrics (Dice, IoU, HD95, ASSD) and HU intensity
metrics (MAE, RMSE, bias). See [../EVALUATION.md](../EVALUATION.md) for how to read
them — in particular why Dice must not be read against 1.0. Run them in this order.

| Script | Purpose |
|---|---|
| `validate_eval_setup.py` | Assert the assumptions the metrics rest on (spacing, axis order, metric maths). Run first; a wrong axis still segments happily. |
| `segment_real_testset.py` | Segment the real test CTs once and cache the labels. Every comparison reuses this. |
| `evaluate_segmentation.py` | The main entry point: segment each run's volumes and score them against the real CT. Repeat `--run` for a view sweep. |
| `summarize_evaluation.py` | Aggregate the per-case CSV into the paper tables. No GPU. |
| `eval_ceiling.py` | What the 128³ grid alone costs the segmenter — the scale Dice should be read against. |
| `eval_data.py` | Shared loaders (module, not an entry point). |

## LIDC-IDRI Dataset Source

For this project, use the **LIDC-IDRI dataset released with the X2CT-GAN GitHub project**. That version already follows the X2CT pipeline, including the bed/table stripping step, so the TF-PRDiT dataloader can load the HDF5 files directly. This avoids subtle mismatches from re-running a different local preprocessing pipeline.

Expected structure after extraction:

```text
<data_root>/
  LIDC-HDF5-256/
    <case_id>/
      ct_xray_data.h5
```

or:

```text
<data_root>/
  LIDC-HDF5-256/
    <case_id>_ct_xray_data.h5
```

The HDF5 files should contain at least:

- `ct`
- `xray1`
- `xray2`

Set the LIDC configs accordingly:

```yaml
data:
  path: "/path/to/data_root"
  target_path: "LIDC-HDF5-256"
```

## Download: X2CT-GAN Preprocessed LIDC-IDRI

Use the archive URL provided by the official X2CT-GAN GitHub dataset instructions:

```bash
python scripts/download_lidc_idri.py \
  --url "PASTE_X2CT_GAN_LIDC_DATASET_ARCHIVE_URL_HERE" \
  --out /path/to/data_root \
  --archive-name LIDC-HDF5-256.zip
```

If the archive already extracts into `LIDC-HDF5-256/`, point `--out` to the parent data directory. If it extracts into another nested folder, move or symlink it so the config path resolves to:

```text
/path/to/data_root/LIDC-HDF5-256
```

Download without extraction:

```bash
python scripts/download_lidc_idri.py \
  --url "PASTE_X2CT_GAN_LIDC_DATASET_ARCHIVE_URL_HERE" \
  --out /path/to/data_root \
  --no-extract
```

## DRR / X-ray Generation

```bash
python scripts/ct2xrays.py --help
```

Common modes include single-view, multi-view, circular, spiral, and dual-view projection generation. The script can auto-detect HU-like input ranges or force normalized-to-HU conversion.

## Notes

- Set final dataset paths in `configs/*.yaml`, not in these scripts.
- Generated data, checkpoints, and `results/` outputs are intentionally not tracked.
