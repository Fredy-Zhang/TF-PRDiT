# Dataset Setup

## LIDC-IDRI

Use the preprocessed HDF5 dataset released with the **X2CT-GAN GitHub
project**. That release follows the X2CT pipeline and includes the bed/table
stripping expected by this codebase.

Download the archive with:

```bash
python scripts/download_lidc_idri.py \
  --url "PASTE_X2CT_GAN_LIDC_DATASET_ARCHIVE_URL_HERE" \
  --out /path/to/data_root \
  --archive-name LIDC-HDF5-256.zip
```

The extracted dataset may use either of these layouts:

```text
/path/to/data_root/
  LIDC-HDF5-256/
    <case_id>/
      ct_xray_data.h5
```

```text
/path/to/data_root/
  LIDC-HDF5-256/
    <case_id>_ct_xray_data.h5
```

Each HDF5 file should contain:

- `ct`
- `xray1`
- `xray2`

Set the dataset location in `configs/lidc_stage2_global.yaml`:

```yaml
data:
  path: "/path/to/data_root"
  target_path: "LIDC-HDF5-256"
```

Run the downloader with `--help` to see all available options:

```bash
python scripts/download_lidc_idri.py --help
```
