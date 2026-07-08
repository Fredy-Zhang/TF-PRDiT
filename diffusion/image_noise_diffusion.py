"""Image-and-Noise diffusion used by training and X-ray guided sampling.

This module contains the complete paper-specific diffusion path:

- ``IaNDiffusion`` is used during training and unconditional sampling.
- ``XrayGuidedIaNDiffusion`` is used by ``sample_xrays.py`` for conditional
  sampling with projection residual guidance.

Both classes share the same forward noising process and training loss. The
model is expected to output two channels per input channel: reconstructed noise
and reconstructed clean image/volume.
"""
import math
import os
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class XrayGuidanceConfig:
    """Controls projection-residual guidance during X-ray conditioned sampling.

    Attributes:
        forward_step: Predictor step multiplier used in guided sampling.
        guidance_scale: Strength of the projection residual gradient.
        save_every: Debug visualization interval in sampling steps.
        rotations: Number of projection rotations used for X-ray rendering.
        noise_weight: Noise scale injected during the correction step.
        save_detailed: Save individual X-ray views and reference volumes.
        debug: Save intermediate debug visualizations when ``sample_dir`` exists.
        guidance_cutoff: Disable guidance at or below this timestep.
    """

    forward_step: int = 4
    guidance_scale: float = 0.70
    save_every: int = 200
    rotations: int = 2
    noise_weight: float = 0.95
    save_detailed: bool = True
    debug: bool = True
    guidance_cutoff: int = 20


def save_individual_xrays(
    xrays: torch.Tensor,
    save_dir: str,
    prefix: str,
) -> None:
    """Save each X-ray projection as an individual grayscale PNG.

    Args:
        xrays: Projection tensor with shape ``[views, H, W]``,
            ``[views, 1, H, W]``, or a batched variant.
        save_dir: Directory where PNG files are written.
        prefix: Filename prefix. View indices are appended automatically.
    """
    import cv2

    os.makedirs(save_dir, exist_ok=True)
    xrays_np = xrays.detach().cpu().numpy()

    if xrays_np.ndim == 4:
        if xrays_np.shape[1] == 1 and xrays_np.shape[0] > 1:
            xrays_np = xrays_np[:, 0, ...]
        elif xrays_np.shape[0] == 1:
            xrays_np = xrays_np[0]
            if xrays_np.ndim == 3 and xrays_np.shape[0] == 1:
                xrays_np = xrays_np[0]
        else:
            xrays_np = xrays_np[0]
    elif xrays_np.ndim == 3 and xrays_np.shape[1] == 1:
        xrays_np = xrays_np.squeeze(1)

    if xrays_np.ndim == 2:
        xrays_np = xrays_np[None, ...]

    for view_idx, img in enumerate(xrays_np):
        if img.ndim == 3:
            img = img.squeeze(0)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)
        cv2.imwrite(
            os.path.join(save_dir, f"{prefix}_view{view_idx:02d}.png"),
            (img * 255).astype(np.uint8),
        )


def save_reference_volume(volume: torch.Tensor, save_dir: str, name: str) -> None:
    """Save a reference CT volume as a NIfTI file.

    Args:
        volume: CT tensor with optional batch/channel dimensions.
        save_dir: Directory where the ``.nii.gz`` file is written.
        name: Output filename stem without extension.
    """
    import nibabel as nib

    os.makedirs(save_dir, exist_ok=True)
    vol_np = volume.detach().cpu().squeeze().numpy()
    nib.save(nib.Nifti1Image(vol_np, np.eye(4)), os.path.join(save_dir, f"{name}.nii.gz"))


class _BaseIaNDiffusion:
    """Shared Image-and-Noise diffusion math.

    The model predicts two tensors at every timestep:
    reconstructed noise and reconstructed clean image/volume.
    """

    def __init__(
        self,
        timestep_respacing: Optional[int] = None,
        loss_type: str = "l2",
        num_timesteps: int = 1000,
        config: Optional[Any] = None,
    ):
        """Initialize shared IaN diffusion settings.

        Args:
            timestep_respacing: Number of sampling timesteps to use. If omitted,
                the full ``num_timesteps`` schedule is used.
            loss_type: Training loss type. Currently only ``"l2"`` is supported.
            num_timesteps: Number of forward diffusion timesteps.
            config: Optional caller-specific config stored for later access.
        """
        self.loss_type = loss_type
        self.num_timesteps = num_timesteps
        self.timestep_respacing = int(timestep_respacing) if timestep_respacing else num_timesteps
        self.config = config
        self._half_pi = math.pi / 2
        self._step_w = math.pi / 2000

    def _timestep_sequence(self) -> range:
        """Return the evenly spaced timestep sequence used for sampling."""
        skip = max(1, self.num_timesteps // self.timestep_respacing)
        return range(0, self.num_timesteps, skip)

    def _broadcast_timesteps(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Reshape 1D timesteps so they broadcast across a volume tensor.

        Args:
            t: Tensor of shape ``[B]`` containing timestep indices.
            x: Volume tensor with batch dimension ``B``.

        Returns:
            ``t`` reshaped to ``[B, 1, 1, 1, 1]`` for 3D volumes.
        """
        return t.reshape(x.shape[:1] + (1,) * (x.ndim - 1)).to(x.device)

    def _predict_noise_and_image(
        self,
        model: Callable,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_kwargs: Dict[str, Any],
        conditions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the model and split its output into noise and image predictions.

        Args:
            model: Callable DiT forward function.
            x_t: Noisy input volume at timestep ``t``.
            t: Timestep tensor.
            model_kwargs: Extra keyword arguments forwarded to the model.
            conditions: Optional conditional tensor for models that accept it.

        Returns:
            Tuple ``(eps_recon, img_recon)`` from the model output channels.
        """
        if conditions is None:
            model_output = model(x_t, t, **model_kwargs)
        else:
            model_output = model(x_t, t, conditions=conditions, **model_kwargs)
        return model_output.chunk(2, dim=1)

    def gen_noise(self, x_start: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
        """Generate Gaussian noise matching ``x_start``.

        Args:
            x_start: Reference tensor whose shape/device/dtype are reused.
            weight: Scalar multiplier for the sampled noise.

        Returns:
            Gaussian noise tensor with the same shape as ``x_start``.
        """
        return torch.randn_like(x_start) * weight

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the IaN forward noising process.

        Args:
            x_start: Clean input volume.
            t: Timestep tensor with shape ``[B]``.
            noise: Optional fixed noise tensor. If omitted, fresh noise is used.

        Returns:
            Noisy volume ``x_t`` formed by cosine/sine interpolation between
            the clean image and Gaussian noise.
        """
        if noise is None:
            noise = self.gen_noise(x_start)

        t_reshape = self._broadcast_timesteps(t, x_start)
        cos_coeff = torch.cos(t_reshape / self.num_timesteps * self._half_pi)
        sin_coeff = torch.sin(t_reshape / self.num_timesteps * self._half_pi)
        return cos_coeff * x_start + sin_coeff * noise

    def training_losses(
        self,
        model: Callable,
        x_start: torch.Tensor,
        t: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        current_epoch: Optional[int] = None,
        y: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the IaN training losses for a batch.

        Args:
            model: Callable DiT forward function.
            x_start: Clean training volumes.
            t: Random timesteps for the batch.
            conditions: Optional model conditions.
            model_kwargs: Extra keyword arguments forwarded to the model.
            current_epoch: Accepted for training-loop compatibility.
            y: Accepted for Gaussian-diffusion API compatibility.

        Returns:
            Dictionary with per-sample ``img_loss`` and ``noise_loss`` tensors.
        """
        if model_kwargs is None:
            model_kwargs = {}

        noise = self.gen_noise(x_start)
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        eps_recon, img_recon = self._predict_noise_and_image(
            model, x_t, t, model_kwargs, conditions=conditions
        )

        if eps_recon.shape != noise.shape:
            raise ValueError(f"Noise prediction shape {eps_recon.shape} does not match {noise.shape}")

        if self.loss_type != "l2":
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented")

        spatial_dims = list(range(1, len(x_start.shape)))
        return {
            "img_loss": (img_recon - x_start).pow(2).mean(dim=spatial_dims),
            "noise_loss": (eps_recon - noise).pow(2).mean(dim=spatial_dims),
        }


class IaNDiffusion(_BaseIaNDiffusion):
    """Core IaN diffusion used by training and unconditional sampling."""

    def p_sample_loop(
        self,
        model: Callable,
        shape: Tuple[int, ...],
        z: torch.Tensor,
        clip_denoised: bool = False,
        progress: bool = False,
        new_sampling: bool = False,
        device: Optional[torch.device] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Sample volumes from initial Gaussian noise.

        Args:
            model: Callable DiT forward function.
            shape: Kept for compatibility with the previous sampling API.
            z: Initial Gaussian noise tensor.
            clip_denoised: Kept for compatibility; not used by IaN sampling.
            progress: Kept for compatibility; progress bars are handled outside.
            new_sampling: Use predictor-corrector sampling when true.
            device: Optional device override. The noise tensor device is used.
            model_kwargs: Extra keyword arguments forwarded to the model.

        Returns:
            Tuple ``(xs, x0_final)`` where ``xs`` contains sampled states and
            ``x0_final`` is the final clean-volume prediction.
        """
        if model_kwargs is None:
            model_kwargs = {}

        seq = self._timestep_sequence()
        if new_sampling:
            print("Using predictor-corrector sampling.")
            x0_preds, xs = self._predictor_corrector_steps(z, seq, model, model_kwargs)
        else:
            print("Using standard gradient-update sampling.")
            x0_preds, xs = self._generalized_steps(z, seq, model, model_kwargs)

        return xs, x0_preds[-1]

    @torch.no_grad()
    def _generalized_steps(
        self,
        x: torch.Tensor,
        seq: range,
        model: Callable,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Run the standard deterministic IaN reverse update.

        Args:
            x: Initial sample tensor.
            seq: Timesteps to traverse in reverse order.
            model: Callable DiT forward function.
            model_kwargs: Extra keyword arguments forwarded to the model.

        Returns:
            Clean-image predictions and intermediate sample states.
        """
        if model_kwargs is None:
            model_kwargs = {}

        n = x.size(0)
        seq_next = [0] + list(seq[:-1])
        x0_preds, xs = [], [x]

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.full((n,), i, device=x.device)
            next_t = torch.full((n,), j, device=x.device)
            at = 1 - (t / self.num_timesteps)[:, None, None, None, None]
            next_at = 1 - (next_t / self.num_timesteps)[:, None, None, None, None]

            xt = xs[-1].to(x.device)
            eps_recon, img_recon = self._predict_noise_and_image(model, xt, t, model_kwargs)
            xt_next = xt - (at - next_at) * self._half_pi * (
                torch.cos(at * self._half_pi) * img_recon
                - torch.sin(at * self._half_pi) * eps_recon
            )

            x0_preds.append(img_recon.cpu())
            xs.append(xt_next.cpu())

        return x0_preds, xs

    @torch.no_grad()
    def _predictor_corrector_steps(
        self,
        x: torch.Tensor,
        seq: range,
        model: Callable,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Run predictor-corrector IaN sampling.

        Args:
            x: Initial sample tensor.
            seq: Timesteps to traverse in reverse order.
            model: Callable DiT forward function.
            model_kwargs: Extra keyword arguments forwarded to the model.

        Returns:
            Clean-image predictions and intermediate sample states.
        """
        if model_kwargs is None:
            model_kwargs = {}

        p = 2
        n = x.size(0)
        seq_next = [0] + list(seq[:-1])
        xs, x0_preds = [x], []
        device = x.device

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.full((n,), i, device=device)
            h = (j - i) * self._step_w

            xt = xs[-1].to(device)
            et, x0_t = self._predict_noise_and_image(model, xt, t, model_kwargs)
            beta_t = (t / self.num_timesteps)[..., None, None, None, None] * self._half_pi
            f_t = torch.sin(beta_t) * x0_t - torch.cos(beta_t) * et

            if i - p * (i - j) > 0:
                xt_pred = xt - p * h * f_t
                t_pred = torch.full((n,), i - p * (i - j), device=device)
                t_corr = torch.full((n,), i - (i - j), device=device)
                beta_corr = (t_corr / self.num_timesteps)[..., None, None, None, None] * self._half_pi
                beta_pred = (t_pred / self.num_timesteps)[..., None, None, None, None] * self._half_pi
                alpha = torch.cos(beta_corr) / torch.cos(beta_pred)
                alpha_noise = torch.sqrt(torch.clamp(1 - alpha**2, min=0.0))
                xt_next = alpha * xt_pred + alpha_noise * self.gen_noise(xt_pred)
            else:
                xt_next = xt - h * f_t

            x0_preds.append(x0_t.cpu())
            xs.append(xt_next.cpu())

        return x0_preds, xs


class XrayGuidedIaNDiffusion(_BaseIaNDiffusion):
    """IaN sampler with X-ray projection residual guidance."""

    def __init__(
        self,
        timestep_respacing: Optional[int] = None,
        loss_type: str = "l2",
        num_timesteps: int = 1000,
        config: Optional[XrayGuidanceConfig] = None,
    ):
        """Initialize the X-ray guided IaN sampler.

        Args:
            timestep_respacing: Number of sampling timesteps.
            loss_type: Training loss type. Currently only ``"l2"`` is supported.
            num_timesteps: Number of forward diffusion timesteps.
            config: Optional ``XrayGuidanceConfig``. Defaults are used when a
                project training config is passed instead.
        """
        guidance_config = config if isinstance(config, XrayGuidanceConfig) else XrayGuidanceConfig()
        super().__init__(
            timestep_respacing=timestep_respacing,
            loss_type=loss_type,
            num_timesteps=num_timesteps,
            config=guidance_config,
        )

    def p_sample_loop(
        self,
        model: Callable,
        z: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
        idx: Optional[int] = None,
        sample_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
        ref_vol: Optional[torch.Tensor] = None,
        new_sampling: bool = False,
        rotations: Optional[int] = None,
        config: Optional[XrayGuidanceConfig] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Run unconditional or X-ray guided sampling.

        Args:
            model: Callable DiT forward function.
            z: Initial Gaussian noise tensor.
            conditions: Optional reference X-ray projections.
            idx: Dataset/sample index used by projection utilities and filenames.
            sample_dir: Optional directory for debug and final X-ray outputs.
            device: Device used for sampling. Defaults to ``z.device``.
            ref_vol: Optional reference CT volume for debug visualizations.
            new_sampling: Use projection-guided predictor-corrector sampling.
            rotations: Optional override for projection rotations.
            config: Optional guidance config override.
            model_kwargs: Extra keyword arguments forwarded to the model.

        Returns:
            Final clean-volume prediction.
        """
        if device is None:
            device = z.device
        if sample_dir is not None:
            os.makedirs(sample_dir, exist_ok=True)

        current_config = config or self.config
        if rotations is not None:
            current_config = replace(current_config, rotations=rotations)

        seq = list(self._timestep_sequence())
        if new_sampling:
            print("Using X-ray guided predictor-corrector sampling.")
            x0_preds, _ = self._guided_steps(
                z, seq, model, conditions, idx, sample_dir, ref_vol, current_config, model_kwargs
            )
        else:
            print("Using standard unguided sampling.")
            x0_preds, _ = self._standard_steps(z, seq, model, model_kwargs)

        return x0_preds[-1]

    @torch.no_grad()
    def _standard_steps(
        self,
        x: torch.Tensor,
        seq: List[int],
        model: Callable,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Run unguided sampling for the X-ray sampler class.

        Args:
            x: Initial sample tensor.
            seq: Timesteps to traverse in reverse order.
            model: Callable DiT forward function.
            model_kwargs: Extra keyword arguments forwarded to the model.

        Returns:
            Clean-image predictions and intermediate sample states.
        """
        if model_kwargs is None:
            model_kwargs = {}

        n = x.size(0)
        seq_next = [0] + list(seq[:-1])
        x0_preds, xs = [], [x]

        t_buffer = torch.empty((n,), device=x.device, dtype=torch.long)
        next_t_buffer = torch.empty((n,), device=x.device, dtype=torch.long)

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t_buffer.fill_(i)
            next_t_buffer.fill_(j)

            at = 1 - (t_buffer / self.num_timesteps)[:, None, None, None, None]
            next_at = 1 - (next_t_buffer / self.num_timesteps)[:, None, None, None, None]

            xt = xs[-1].to(x.device)
            eps_recon, img_recon = self._predict_noise_and_image(model, xt, t_buffer, model_kwargs)
            img_recon = torch.clamp(img_recon, -1, 1)
            xt_next = xt - (at - next_at) * self._half_pi * (
                torch.cos(at * self._half_pi) * img_recon
                - torch.sin(at * self._half_pi) * eps_recon
            )

            x0_preds.append(img_recon.detach().cpu())
            xs.append(xt_next.detach().cpu())

        return x0_preds, xs

    def _guided_steps(
        self,
        x: torch.Tensor,
        seq: List[int],
        model: Callable,
        conditions: Optional[torch.Tensor],
        idx: Optional[int],
        sample_dir: Optional[str],
        ref_vol: Optional[torch.Tensor],
        config: XrayGuidanceConfig,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Run X-ray projection-guided predictor-corrector sampling.

        Args:
            x: Initial sample tensor.
            seq: Timesteps to traverse in reverse order.
            model: Callable DiT forward function.
            conditions: Reference X-ray projections used for residual guidance.
            idx: Dataset/sample index used by projection utilities and filenames.
            sample_dir: Optional output directory for debug visualizations.
            ref_vol: Optional reference CT volume for debug visualizations.
            config: Guidance hyperparameters.
            model_kwargs: Extra keyword arguments forwarded to the model.

        Returns:
            Clean-image predictions and intermediate sample states.
        """
        from conds.ct2xrays import get_xrays_from_ct
        from conds.utils import get_residual, save_comparison_grid, save_middle_slices

        if model_kwargs is None:
            model_kwargs = {}

        n = x.size(0)
        seq_next = [0] + list(seq[:-1])
        xs, x0_preds = [x], []
        device = x.device
        step_times = []
        step_counter = 0
        guidance_disabled_logged = False

        cond_tensor = None
        if conditions is not None:
            print(f"[+] Input conditions shape: {conditions.shape}")
            cond_stats = conditions.cpu()
            print(
                f"[+] Conditions stats: max={cond_stats.max().item():.5f}, "
                f"min={cond_stats.min().item():.5f}, mean={cond_stats.mean().item():.5f}, "
                f"std={cond_stats.std().item():.5f}"
            )
            cond_tensor = conditions.to(device)
            if sample_dir:
                os.makedirs(os.path.join(sample_dir, "gen"), exist_ok=True)

        t_start_total = time.perf_counter()
        t_buffer = torch.empty((n,), device=device, dtype=torch.long)

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t0 = time.perf_counter()
            t_buffer.fill_(i)
            h = (j - i) * self._step_w
            step_size = i - j
            xt = xs[-1].to(device)

            apply_guidance = cond_tensor is not None and i > config.guidance_cutoff
            if cond_tensor is not None and not apply_guidance and not guidance_disabled_logged:
                print(f"[INFO] Step {i}: guidance disabled for t <= {config.guidance_cutoff}.")
                guidance_disabled_logged = True

            if apply_guidance:
                with torch.set_grad_enabled(True):
                    xt.requires_grad_(True)
                    et, x0_t = self._predict_noise_and_image(model, xt, t_buffer, model_kwargs)
                    sinos_hat = get_xrays_from_ct(
                        x0_t,
                        idx=idx,
                        device=device,
                        rotations=config.rotations,
                    )
                    residual = get_residual(sinos_hat, cond_tensor)
                    residual_norm = torch.linalg.norm(residual)
                    if residual_norm > 0:
                        norm_grad = torch.autograd.grad(residual_norm, x0_t, retain_graph=True)[0]
                    else:
                        norm_grad = torch.zeros_like(xt)

                    if sample_dir and config.debug and step_counter % config.save_every == 0:
                        gen_dir = os.path.join(sample_dir, "gen")
                        save_middle_slices(
                            x0_t.detach().cpu(),
                            f"{i}_{idx}",
                            save_dir=gen_dir,
                            ref_volume=ref_vol,
                        )
                        save_comparison_grid(
                            sinos_hat.detach().cpu(),
                            conditions.cpu(),
                            f"{i}_{idx}",
                            save_dir=gen_dir,
                        )
                        print(
                            f"[+] Step {i}: Loss={residual_norm.item():.6f}, "
                            f"Norm grad norm={norm_grad.norm().item():.6f}, "
                            f"rho={config.guidance_scale:.6f}"
                        )
                        if config.save_detailed:
                            self._save_guidance_debug_outputs(
                                sample_dir, sinos_hat, conditions, ref_vol, idx, i
                            )
            else:
                with torch.no_grad():
                    et, x0_t = self._predict_noise_and_image(model, xt, t_buffer, model_kwargs)
                    norm_grad = None

            beta_t = (t_buffer / self.num_timesteps)[..., None, None, None, None] * self._half_pi
            f_t = torch.sin(beta_t) * x0_t - torch.cos(beta_t) * et

            if (i - config.forward_step * step_size) >= 0 and i != 0:
                if apply_guidance:
                    t_alpha = i / self.num_timesteps
                    decay_scale = 0.5 * (1 + math.cos((1 - t_alpha) * math.pi))
                    xt_pred = xt - config.forward_step * h * f_t - config.guidance_scale * decay_scale * norm_grad
                else:
                    xt_pred = xt - config.forward_step * h * f_t

                t_pred = torch.full((n,), i - config.forward_step * (i - j), device=device)
                t_corr = torch.full((n,), i - (i - j), device=device)
                beta_corr = (t_corr / self.num_timesteps)[..., None, None, None, None] * self._half_pi
                beta_pred = (t_pred / self.num_timesteps)[..., None, None, None, None] * self._half_pi
                alpha = torch.cos(beta_corr) / torch.cos(beta_pred)
                alpha_noise = torch.sqrt(torch.clamp(1 - alpha**2, min=0.0))
                xt_next = alpha * xt_pred + alpha_noise * self.gen_noise(
                    xt_pred,
                    weight=config.noise_weight,
                )
            else:
                xt_next = xt - h * f_t

            x0_preds.append(x0_t.detach().cpu())
            xs.append(xt_next.detach().cpu())
            step_counter += 1
            step_times.append(time.perf_counter() - t0)

        self._log_sampling_time(step_times, t_start_total)
        if sample_dir and x0_preds:
            self._save_final_xrays(x0_preds[-1].to(device), sample_dir, idx, device, config)

        return x0_preds, xs

    def _save_guidance_debug_outputs(
        self,
        sample_dir: str,
        sinos_hat: torch.Tensor,
        conditions: torch.Tensor,
        ref_vol: Optional[torch.Tensor],
        idx: Optional[int],
        timestep: int,
    ) -> None:
        """Save intermediate X-ray guidance debug artifacts.

        Args:
            sample_dir: Root directory for the current sample outputs.
            sinos_hat: Generated X-ray projections from the current CT estimate.
            conditions: Reference X-ray projections.
            ref_vol: Optional reference CT volume.
            idx: Dataset/sample index used in filenames.
            timestep: Current diffusion timestep used in filenames.
        """
        gen_dir = os.path.join(sample_dir, "gen")
        xray_save_dir = os.path.join(gen_dir, "xray_views")
        save_individual_xrays(
            sinos_hat.detach().cpu(),
            os.path.join(xray_save_dir, "estimated"),
            prefix=f"{timestep}_{idx}",
        )
        save_individual_xrays(
            conditions.cpu(),
            os.path.join(xray_save_dir, "reference"),
            prefix=f"{timestep}_{idx}",
        )
        if ref_vol is not None:
            save_reference_volume(
                ref_vol,
                os.path.join(sample_dir, "reference"),
                name=f"{timestep}_{idx}_reference_volume",
            )

    def _save_final_xrays(
        self,
        final_ct: torch.Tensor,
        sample_dir: str,
        idx: Optional[int],
        device: torch.device,
        config: XrayGuidanceConfig,
    ) -> None:
        """Render and save final X-ray projections from the generated CT.

        Args:
            final_ct: Final generated CT tensor.
            sample_dir: Root directory for the current sample outputs.
            idx: Dataset/sample index used in filenames.
            device: Device used by the projection renderer.
            config: Guidance config containing the projection rotation count.
        """
        from conds.ct2xrays import get_xrays_from_ct

        final_xrays = get_xrays_from_ct(
            final_ct,
            idx=idx,
            device=device,
            rotations=config.rotations,
        )
        final_xray_dir = os.path.join(sample_dir, "final_xrays")
        save_individual_xrays(
            final_xrays.detach().cpu(),
            final_xray_dir,
            prefix=f"final_{idx if idx is not None else 0}",
        )
        print(f"[+] Final x-rays saved to: {final_xray_dir}")

    def _log_sampling_time(self, step_times: List[float], start_time: float) -> None:
        """Print total and average sampling time.

        Args:
            step_times: Per-step durations in seconds.
            start_time: Wall-clock timestamp captured before sampling started.
        """
        if not step_times:
            return
        avg_step_time = sum(step_times) / len(step_times)
        total_time = time.perf_counter() - start_time
        print(f"[+] Mean per-step time: {avg_step_time * 1000:.2f} ms over {len(step_times)} steps")
        print(f"[+] Total sampling time: {total_time:.2f} s")
