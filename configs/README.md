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
  --ckpt pretrained/tf_prdit_lidc.pt \
  --rotations 2 \
  --output-dir outputs_Cond \
  --new
```

Use `--rotations` to choose the number of conditioning X-ray views.

## Downstream Task Configs

The downstream config templates keep the same pretrained LIDC prior and add an explicit `downstream` operator section:

| Config | Task |
|---|---|
| `configs/lidc_downstream_super_resolution.yaml` | Volumetric super-resolution |
| `configs/lidc_downstream_infilling.yaml` | Volumetric infilling |
| `configs/lidc_downstream_deblurring.yaml` | Volumetric deblurring |

These configs define operator settings such as super-resolution scale, infilling mask, and blur kernel. They are intended for the downstream sampling path that uses the same frozen prior with a different measurement operator.
