# Diffusion Module

This folder contains the paper-specific Image-and-Noise diffusion code.

## Files

- `image_noise_diffusion.py`: paper-specific Image-and-Noise diffusion.
  - `IaNDiffusion`: core training and unconditional sampling.
  - `XrayGuidedIaNDiffusion`: X-ray conditional sampling with projection residual guidance.

- `__init__.py`: factory used by `train.py` to create `IaNDiffusion`.

## Public API

- `diffusion.loading_diffusion(config, rank=0)`: validates the config and returns
  the training diffusion object.
- `diffusion.image_noise_diffusion.IaNDiffusion`: computes IaN training losses
  and unconditional samples.
- `diffusion.image_noise_diffusion.XrayGuidedIaNDiffusion`: runs the X-ray
  conditional sampler used by `sample_xrays.py`.
- `diffusion.image_noise_diffusion.XrayGuidanceConfig`: sampling/debug settings
  for projection residual guidance.

For this project, the main training path is:

```text
train.py -> diffusion.loading_diffusion() -> image_noise_diffusion.IaNDiffusion
```

The X-ray conditional sampling path is:

```text
sample_xrays.py -> image_noise_diffusion.XrayGuidedIaNDiffusion
```
