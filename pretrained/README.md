# Pretrained Weights

Put released TF-PRDiT checkpoint files in this directory. Weight files are large and should not be committed to git.

Expected layout:

```text
pretrained/
  tf_prdit_lidc.pt
```

## Download

After the checkpoint release URLs are available, download them into this folder:

```bash
mkdir -p pretrained

curl -L "<TF_PRDIT_LIDC_CHECKPOINT_URL>" \
  -o pretrained/tf_prdit_lidc.pt
```

If the weights are hosted on Google Drive, install `gdown` and download with:

```bash
pip install gdown

gdown --fuzzy "<TF_PRDIT_LIDC_GOOGLE_DRIVE_URL>" \
  -O pretrained/tf_prdit_lidc.pt
```

## Use

For X-ray-guided sampling, pass the checkpoint with `--ckpt`:

```bash
python sample_xrays.py \
  --config lidc_stage2_global.yaml \
  --ckpt pretrained/tf_prdit_lidc.pt \
  --num-samples 100 \
  --rotations 2 \
  --output-dir outputs_Cond
```
