# Pretrained Weights

Put released TF-PRDiT checkpoint files in this directory. Weight files are large and should not be committed to git.

Expected layout:

```text
pretrained/
  lidc_stage2_global.pt
```

## Download

After the checkpoint release URLs are available, download them into this folder:

```bash
mkdir -p pretrained

curl -L "<LIDC_STAGE2_GLOBAL_CHECKPOINT_URL>" \
  -o pretrained/lidc_stage2_global.pt
```

If the weights are hosted on Google Drive, install `gdown` and download with:

```bash
pip install gdown

gdown --fuzzy "<LIDC_STAGE2_GLOBAL_GOOGLE_DRIVE_URL>" \
  -O pretrained/lidc_stage2_global.pt
```

## Use

For X-ray-guided sampling, pass the stage-2/global checkpoint with `--ckpt`:

```bash
python sample_xrays.py \
  --config lidc_stage2_global.yaml \
  --ckpt pretrained/lidc_stage2_global.pt \
  --num-samples 100 \
  --rotations 2 \
  --output-dir outputs_Cond \
  --new
```
