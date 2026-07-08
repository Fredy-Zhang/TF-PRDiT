# Configs

For conditional sampling with released weights, use:

```text
configs/lidc_stage2_global.yaml
```

Set the LIDC dataset location in that config:

```yaml
data:
  path: "/path/to/data_root"
  target_path: "LIDC-HDF5-256"
```

Then pass the config to `sample_xrays.py`:

```bash
python sample_xrays.py \
  --config lidc_stage2_global.yaml \
  --ckpt pretrained/lidc_stage2_global.pt \
  --rotations 2 \
  --output-dir outputs_Cond \
  --new
```

Use `--rotations` to choose the number of conditioning X-ray views.
