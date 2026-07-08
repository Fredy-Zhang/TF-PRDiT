# TF-PRDiT

Official code for **From Sparse X-rays to 3D CT: Training-Free Reconstruction with Diffusion Priors**.

TF-PRDiT uses a pretrained 3D diffusion prior for sparse X-ray-to-CT reconstruction. This repository README is focused on the inference workflow: download the dataset, download pretrained weights, and run conditional sampling with different numbers of X-ray views.

Paper: [arXiv:2606.20763](https://arxiv.org/abs/2606.20763)

<p align="center">
  <img src="assets/x2ct.png" alt="X-ray-to-CT reconstruction results across axial, coronal, and sagittal views." width="920">
</p>

## Model Structure

TF-PRDiT keeps a pretrained 3D DiT prior frozen during conditional sampling. For each reverse diffusion step, the sampler projects the current denoised CT estimate into X-ray space, compares it with the input sparse X-rays, and uses that consistency signal to guide the next CT estimate.

<p align="center">
  <img src="assets/overview.png" alt="TF-PRDiT model structure: frozen 3D prior, predictor-corrector sampling, and X-ray consistency guidance." width="920">
</p>

The same sampling interface can guide the prior with different measurement operators. This README focuses on sparse X-ray-to-CT sampling.

<p align="center">
  <img src="assets/inverse_problem.png" alt="TF-PRDiT inverse problem structure across sparse X-ray, super-resolution, infilling, and deblurring operators." width="920">
</p>

## Downstream Task Configs

In addition to sparse X-ray-to-CT sampling, the same frozen prior can be configured for volumetric super-resolution, infilling, and deblurring experiments. The downstream YAML files keep the same LIDC prior/data settings and add a `downstream` section describing the measurement operator.

| Task | Config | Figure |
|---|---|---|
| Super-resolution | `configs/lidc_downstream_super_resolution.yaml` | `assets/sr.png` |
| Infilling | `configs/lidc_downstream_infilling.yaml` | `assets/infilling.png` |
| Deblurring | `configs/lidc_downstream_deblurring.yaml` | `assets/deblurr.png` |

| Super-resolution | Infilling | Deblurring |
|---|---|---|
| <img src="assets/sr.png" alt="3D CT super-resolution result." width="280"> | <img src="assets/infilling.png" alt="3D CT infilling result." width="280"> | <img src="assets/deblurr.png" alt="3D CT deblurring result." width="280"> |

## Repository Layout

```text
configs/                  Sampling config files
datasets/                 Dataset loaders
diffusion/                Image-and-Noise diffusion and X-ray guided sampler
models/                   3D DiT model definitions
pretrained/               Pretrained checkpoint download instructions
scripts/                  Dataset download and utility scripts
sample_xrays.py           Sparse X-ray-to-CT conditional sampling
utils/download.py         Checkpoint loading helper
```

## Download Dataset

For LIDC-IDRI, use the preprocessed HDF5 dataset released with the **X2CT-GAN GitHub project**. That release already follows the X2CT pipeline and includes the bed/table stripping expected by this codebase.

Download the archive with:

```bash
python scripts/download_lidc_idri.py \
  --url "PASTE_X2CT_GAN_LIDC_DATASET_ARCHIVE_URL_HERE" \
  --out /path/to/data_root \
  --archive-name LIDC-HDF5-256.zip
```

Expected extracted layout:

```text
/path/to/data_root/
  LIDC-HDF5-256/
    <case_id>/
      ct_xray_data.h5
```

or:

```text
/path/to/data_root/
  LIDC-HDF5-256/
    <case_id>_ct_xray_data.h5
```

The HDF5 files should contain:

- `ct`
- `xray1`
- `xray2`

Then set the dataset location in the LIDC configs:

```yaml
data:
  path: "/path/to/data_root"
  target_path: "LIDC-HDF5-256"
```

Relevant config files:

- `configs/lidc_stage2_global.yaml`

## Download Pretrained Weights

Place the released TF-PRDiT sampling checkpoint in `pretrained/`.

```bash
mkdir -p pretrained

curl -L "<TF_PRDIT_LIDC_CHECKPOINT_URL>" \
  -o pretrained/tf_prdit_lidc.pt
```

If the checkpoint is hosted on Google Drive:

```bash
pip install gdown

gdown --fuzzy "<TF_PRDIT_LIDC_GOOGLE_DRIVE_URL>" \
  -O pretrained/tf_prdit_lidc.pt
```

See [pretrained/README.md](pretrained/README.md) for the checkpoint folder convention.

## Conditional Sampling

Run sparse X-ray-to-CT reconstruction with:

```bash
python sample_xrays.py \
  --config lidc_stage2_global.yaml \
  --ckpt pretrained/tf_prdit_lidc.pt \
  --num-samples 100 \
  --num-sampling-steps 1000 \
  --rotations 2 \
  --output-dir outputs_Cond \
  --new
```

Use metrics-only mode to avoid saving intermediate PNG/NIfTI files:

```bash
python sample_xrays.py \
  --config lidc_stage2_global.yaml \
  --ckpt pretrained/tf_prdit_lidc.pt \
  --num-samples 100 \
  --rotations 2 \
  --output-dir outputs_metrics \
  --new \
  --no-save-intermediate
```

## Change X-ray View Number

The number of input X-ray views is controlled by `--rotations`.

```bash
# One X-ray view
python sample_xrays.py \
  --config lidc_stage2_global.yaml \
  --ckpt pretrained/tf_prdit_lidc.pt \
  --num-samples 10 \
  --rotations 1 \
  --output-dir outputs_1view \
  --new

# Two orthogonal X-ray views, default paper-style setting
python sample_xrays.py \
  --config lidc_stage2_global.yaml \
  --ckpt pretrained/tf_prdit_lidc.pt \
  --num-samples 10 \
  --rotations 2 \
  --output-dir outputs_2views \
  --new

# Four X-ray views
python sample_xrays.py \
  --config lidc_stage2_global.yaml \
  --ckpt pretrained/tf_prdit_lidc.pt \
  --num-samples 10 \
  --rotations 4 \
  --output-dir outputs_4views \
  --new
```

View selection follows `conds/ct2xrays.py`:

- `--rotations 1`: one frontal view.
- `--rotations 2`: two orthogonal views at `0` and `90` degrees.
- `--rotations N` where `N > 2`: keeps `0` and `90` degrees and fills the remaining views across the rotation range.

The current sampler generates DRR/X-ray conditions from the CT volume in the dataset for each sample. To change the number of conditioning X-rays, change `--rotations`.

## Useful Commands

```bash
python scripts/download_lidc_idri.py --help
python sample_xrays.py --help
python scripts/ct2xrays.py --help
```

## Citation

```bibtex
@misc{zhang2026tfprdit,
  title={From Sparse X-rays to 3D CT: Training-Free Reconstruction with Diffusion Priors},
  author={Zhang, Zhenkai and Hiller, Markus and Ehinger, Krista A. and Drummond, Tom},
  year={2026},
  eprint={2606.20763},
  archivePrefix={arXiv},
  primaryClass={eess.IV}
}
```
