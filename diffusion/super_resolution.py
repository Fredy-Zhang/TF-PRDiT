"""Image and Noise (IaN) Diffusion for Super-Resolution Tasks."""
import math
import os
import time
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

import numpy as np
import torch as th
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from conds.utils import save_middle_slices


def _save_single(data: np.ndarray, path: str, cmap: str = 'gray', colorbar: bool = False) -> None:
    """Save a single 2D slice as an individual image."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    im = ax.imshow(data, cmap=cmap); ax.axis('off')
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches='tight', pad_inches=0.02 if colorbar else 0, dpi=150)
    plt.close()


def save_sr_comparison(
    estimated: th.Tensor,
    conditions: th.Tensor,
    ref_volume: Optional[th.Tensor],
    step: str,
    save_dir: str = "intermediate_results",
) -> None:
    """
    Save a 3-way (or 4-column) comparison of middle slices:
      Estimated | Conditions (upsampled) | Reference | Difference (est vs ref)
    
    Args:
        estimated:  [B, C, D, H, W] or [C, D, H, W] — predicted high-res volume.
        conditions: [C, D_low, H_low, W_low] or [B, C, D_low, H_low, W_low] — low-res input.
        ref_volume: [D, H, W] or [B, C, D, H, W] — ground-truth high-res volume (optional).
        step:       Label for the current step.
        save_dir:   Directory to save the figure.
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- Extract estimated as numpy [D, H, W] ---------------------------------
    est = estimated.detach().cpu().float()
    if est.ndim == 5:
        est = est[0, 0]
    elif est.ndim == 4:
        est = est[0]
    est_np = est.numpy()
    D, H, W = est_np.shape

    # --- Upsample conditions to high-res size for visual comparison -----------
    cond = conditions.detach().cpu().float()
    if cond.ndim == 3:
        cond = cond.unsqueeze(0)  # [1, D_low, H_low, W_low]
    if cond.ndim == 4:
        cond = cond.unsqueeze(0)  # [1, 1, D_low, H_low, W_low]
    cond_up = F.interpolate(cond, size=(D, H, W), mode='trilinear', align_corners=False)
    cond_np = cond_up[0, 0].numpy()

    # --- Extract reference as numpy [D, H, W] --------------------------------
    ref_np = None
    if ref_volume is not None:
        ref = ref_volume.detach().cpu().float()
        if ref.ndim == 5:
            ref_np = ref[0, 0].numpy()
        elif ref.ndim == 3:
            ref_np = ref.numpy()
        elif ref.ndim == 4:
            ref_np = ref[0].numpy()

    # --- Helper: normalise a 2-D slice to [0, 1] ----------------------------
    def _norm(s):
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo + 1e-8)

    # --- Extract middle slices ------------------------------------------------
    view_names = ['Sagittal', 'Coronal', 'Axial']
    est_slices  = [est_np[:, :, W // 2], est_np[:, H // 2, :], est_np[D // 2, :, :]]
    cond_slices = [cond_np[:, :, W // 2], cond_np[:, H // 2, :], cond_np[D // 2, :, :]]

    if ref_np is not None:
        ref_slices = [ref_np[:, :, W // 2], ref_np[:, H // 2, :], ref_np[D // 2, :, :]]
        ncols = 4   # estimated | conditions | reference | difference
    else:
        ref_slices = [None, None, None]
        ncols = 2   # estimated | conditions

    fig, axes = plt.subplots(3, ncols, figsize=(5 * ncols, 15))

    for row, (vname, e_s, c_s, r_s) in enumerate(
        zip(view_names, est_slices, cond_slices, ref_slices)
    ):
        e_s = _norm(e_s)
        c_s = _norm(c_s)

        axes[row, 0].imshow(e_s, cmap='gray')
        axes[row, 0].set_title(f'Estimated {vname} (t={step})')

        axes[row, 1].imshow(c_s, cmap='gray')
        axes[row, 1].set_title(f'Condition {vname} (upsampled)')

        if ref_np is not None:
            r_s = _norm(r_s)
            axes[row, 2].imshow(r_s, cmap='gray')
            axes[row, 2].set_title(f'Reference {vname}')

            diff = np.abs(e_s - r_s)
            im = axes[row, 3].imshow(diff, cmap='hot')
            axes[row, 3].set_title('Difference (Est vs Ref)')
            plt.colorbar(im, ax=axes[row, 3])

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'sr_comparison_step_{step}.png'))
    plt.close()

    # Save individual slices
    indiv_dir = os.path.join(save_dir, f'individual_{step}')
    os.makedirs(indiv_dir, exist_ok=True)
    view_lower = ['sagittal', 'coronal', 'axial']
    for vl, e_s, c_s, r_s in zip(view_lower, est_slices, cond_slices, ref_slices):
        e_s, c_s = _norm(e_s), _norm(c_s)
        _save_single(e_s, os.path.join(indiv_dir, f'estimated_{vl}.png'), cmap='gray')
        _save_single(c_s, os.path.join(indiv_dir, f'lowres_upsampled_{vl}.png'), cmap='gray')
        if r_s is not None:
            r_s = _norm(r_s)
            _save_single(r_s, os.path.join(indiv_dir, f'reference_{vl}.png'), cmap='gray')
            _save_single(np.abs(e_s - r_s), os.path.join(indiv_dir, f'difference_{vl}.png'), cmap='hot', colorbar=True)

@dataclass
class DiffusionConfig:
    """Configuration for IaN Diffusion super-resolution."""
    forward_step: int = 2
    backward_step: int = 1
    guidance_scale: float = 20.0
    save_every: int = 100
    noise_weight: float = 0.9
    save_detailed: bool = True
    debug: bool = True
    
    # Super-resolution parameters
    sr_scale: int = 4              # Downsampling scale factor
    sr_cutoff: int = 100           # Timestep cutoff for SR guidance


class ConditionalGuidance:
    @staticmethod
    def compute_sr_guidance(
        x0_t: th.Tensor,
        low_res: th.Tensor,
        scale: int
    ) -> th.Tensor:
        """
        Super-resolution guidance: downsampled prediction should match low-res input.
        
        Args:
            x0_t: Predicted high-res volume [B, C, D, H, W]
            low_res: Low-resolution input [C, D_low, H_low, W_low]
            scale: Downsampling scale factor
        
        Returns:
            loss: MSE between downsampled prediction and low-res input
        """
        # Downsample predicted high-res to match low-res size
        x0_t_down = F.interpolate(
            x0_t,
            size=low_res.shape[-3:],  # Match spatial dims of low_res
            mode='trilinear',
            align_corners=False
        )
        
        # Ensure low_res has batch dim for comparison
        low_res_batch = low_res.unsqueeze(0) if low_res.ndim == 4 else low_res
        
        diff = x0_t_down - low_res_batch
        loss = th.linalg.norm(diff)
        return loss


class IaNDiffusion:
    """Image and Noise (IaN) Diffusion model for super-resolution tasks."""
    
    def __init__(
        self,
        timestep_respacing: Optional[int] = None,
        loss_type: str = "l2",
        num_timesteps: int = 1000,
        config: Optional[DiffusionConfig] = None,
    ):
        """Initialize IaN Diffusion model for super-resolution."""
        self.loss_type = loss_type
        self.num_timesteps = num_timesteps
        self.timestep_respacing = int(timestep_respacing) if timestep_respacing else num_timesteps
        self.config = DiffusionConfig()
        self.data_config = config
        
        # Precompute constants for efficiency
        self._half_pi = math.pi / 2
        self._step_w = math.pi / 2000
    
    def gen_noise(self, x_start: th.Tensor, weight: float = 1.0) -> th.Tensor:
        """Generate noise for the diffusion model."""
        return th.randn_like(x_start) * weight
    
    def q_sample(self, x_start: th.Tensor, t: th.Tensor) -> th.Tensor:
        """Apply forward diffusion process."""
        noise = self.gen_noise(x_start)
        t_reshape = t.reshape(x_start.shape[:1] + (1,) * (x_start.ndim - 1)).to(x_start.device)
        
        cos_coeff = th.cos(t_reshape / self.num_timesteps * self._half_pi)
        sin_coeff = th.sin(t_reshape / self.num_timesteps * self._half_pi)
        
        return cos_coeff * x_start + sin_coeff * noise
    
    def p_sample_loop(
        self,
        model: th.nn.Module,
        z: th.Tensor,
        conditions: Optional[th.Tensor] = None,  # Low-res image
        idx: Optional[int] = None,
        sample_dir: Optional[str] = None,
        device: Optional[th.device] = None,
        ref_vol: Optional[th.Tensor] = None,
        new_sampling: bool = False,
        config: Optional[DiffusionConfig] = None,
        model_kwargs: Optional[Dict[str, Any]] = None
    ) -> th.Tensor:
        """
        Super-resolution: generate high-res volume consistent with low-res input.
        
        Args:
            conditions: Low-resolution image [C, D_low, H_low, W_low]
        """
        if device is None:
            device = next(model.parameters()).device
        if sample_dir is None:
            sample_dir = "./"
        os.makedirs(sample_dir, exist_ok=True)
        
        current_config = config or self.config

        skip = self.num_timesteps // self.timestep_respacing
        seq = list(range(0, self.num_timesteps, skip))
        
        print("="*30)
        x0_preds, xs = self._generalized_steps_optimized(
            z, seq, model, 
            conditions=conditions,
            idx=idx, 
            sample_dir=sample_dir, 
            ref_vol=ref_vol,
            config=current_config,
            model_kwargs=model_kwargs
        )
        
        return x0_preds[-1]
    
    def _generalized_steps_optimized(
        self,
        x: th.Tensor,
        seq: List[int],
        model: th.nn.Module,
        conditions: Optional[th.Tensor] = None,
        idx: Optional[int] = None,
        sample_dir: Optional[str] = None,
        ref_vol: Optional[th.Tensor] = None,
        config: Optional[DiffusionConfig] = None,
        model_kwargs: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[th.Tensor], List[th.Tensor]]:

        if model_kwargs is None:
            model_kwargs = {}
        
        config = self.config
        n = x.size(0)
        p = config.forward_step
        seq_next = [0] + list(seq[:-1])
        xs, x0_preds = [x], []
        device = x.device
        step_counter = 0
        guidance_disabled_logged = False
        
        # Prepare super-resolution conditions
        cond_tensor = None
        if conditions is not None:
            cond_tensor = conditions.to(device)
            print(f"[+] Super-resolution: low-res shape = {cond_tensor.shape}")
            print(f"[+] Target high-res shape = {x.shape}")
            print(f"[+] Scale factor = {config.sr_scale}")
            if sample_dir:
                os.makedirs(os.path.join(sample_dir, "sr"), exist_ok=True)
        
        # Initialize guidance handler
        guidance_handler = ConditionalGuidance()

        step_times = []
        t_start_total = time.perf_counter()
        
        t_buffer = th.empty((n,), device=device, dtype=th.long)
        
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t0 = time.perf_counter()
            
            t_buffer.fill_(i)
            h = (j - i) * self._step_w  # negative scalar
            step_size = i - j
            xt = xs[-1].to(device)
            
            # Determine whether to apply guidance
            apply_guidance = cond_tensor is not None and i > config.sr_cutoff

            if cond_tensor is not None and not apply_guidance and not guidance_disabled_logged:
                print(f"[INFO] Step {i}: SR guidance disabled (t <= {config.sr_cutoff}).")
                guidance_disabled_logged = True
            
            if apply_guidance:
                with th.set_grad_enabled(True):
                    xt.requires_grad_(True)
                    et, x0_t = model(xt, t_buffer).chunk(2, dim=1)
                    
                    norm_sr_loss = guidance_handler.compute_sr_guidance(x0_t, 
                                                                        cond_tensor, 
                                                                        config.sr_scale)
                    
                    if norm_sr_loss > 0:
                        norm_grad = th.autograd.grad(
                            norm_sr_loss, x0_t,
                            retain_graph=True
                        )[0]
                    else:
                        norm_grad = th.zeros_like(x0_t)
                    
                    # Logging and visualization
                    if config.debug and step_counter % config.save_every == 0:
                        print(f"[+] Step {i}: SR Loss={norm_sr_loss.item():.6f}, Grad norm={norm_grad.norm().item():.6f}")
                        
                        gen_dir = os.path.join(sample_dir, "gen")
                        
                        # Save predicted high-res slices (est vs ref)
                        save_middle_slices(
                            x0_t.detach().cpu(), f"{i}_{idx}", 
                            save_dir=gen_dir, 
                            ref_volume=ref_vol
                        )
                        
                        # Save 3-way comparison: estimated | conditions (upsampled) | reference
                        sr_dir = os.path.join(sample_dir, "sr")
                        save_sr_comparison(
                            estimated=x0_t.detach(),
                            conditions=cond_tensor,
                            ref_volume=ref_vol,
                            step=f"{i}_{idx}",
                            save_dir=sr_dir,
                        )
                        
                        # Save norm_grad slices
                        grad_dir = os.path.join(sample_dir, "grad")
                        os.makedirs(grad_dir, exist_ok=True)
                        g = norm_grad.detach().cpu().float()
                        if g.ndim == 5:
                            g = g[0, 0]
                        elif g.ndim == 4:
                            g = g[0]
                        g_np = g.numpy()
                        gD, gH, gW = g_np.shape
                        for vl, slc in zip(
                            ['sagittal', 'coronal', 'axial'],
                            [g_np[:, :, gW // 2], g_np[:, gH // 2, :], g_np[gD // 2, :, :]],
                        ):
                            _save_single(slc, os.path.join(grad_dir, f'grad_{vl}_{i}_{idx}.png'),
                                         cmap='hot', colorbar=True)

            else:
                # No guidance path
                with th.no_grad():
                    et, x0_t = model(xt, t_buffer, **model_kwargs).chunk(2, dim=1)
                    norm_grad = None
            
            # Compute drift term
            beta_t = (t_buffer / self.num_timesteps)[..., None, None, None, None] * self._half_pi
            f_t = th.sin(beta_t) * x0_t - th.cos(beta_t) * et

            # Update step with predictor-corrector
            if (i - p * step_size) >= 0 and i != 0:
                if not apply_guidance:
                    xt_pred = xt - p * h * f_t
                else:
                    # Compute adaptive guidance scale
                    # t_alpha = i / self.num_timesteps
                    # decay_scale = 0.5 * (1 + math.cos((1 - t_alpha) * math.pi))
                    rho = config.guidance_scale #* decay_scale
                        
                    xt_pred = xt - p * h * f_t - rho * norm_grad
                
                t_pred = th.full((n,), i - p * (i - j), device=device)
                t_corr = th.full((n,), i - (i - j), device=device)
                
                beta_corr = (t_corr / self.num_timesteps)[..., None, None, None, None] * self._half_pi
                beta_pred = (t_pred / self.num_timesteps)[..., None, None, None, None] * self._half_pi
                
                alpha = th.cos(beta_corr) / th.cos(beta_pred)
                alpha_1 = th.sqrt(th.clamp(1 - alpha**2, min=0.0))
                
                xt_next = alpha * xt_pred + alpha_1 * self.gen_noise(xt_pred, weight=config.noise_weight)
            else:
                xt_next = xt - h * f_t

            x0_preds.append(x0_t.detach().cpu())
            xs.append(xt_next.detach().cpu())
            
            step_counter += 1
            t1 = time.perf_counter()
            dt = t1 - t0
            step_times.append(dt)

        total_time = time.perf_counter() - t_start_total
        avg_step_time = sum(step_times) / len(step_times)
        print(f"[+] Mean per-step time: {avg_step_time*1000:.2f} ms over {len(step_times)} steps")
        print(f"[+] Total sampling time: {total_time:.2f} s")

        # Save final super-resolved result
        if sample_dir and len(x0_preds) > 0:
            final_ct = x0_preds[-1].to(device)
            final_sr_dir = os.path.join(sample_dir, "final_sr")
            os.makedirs(final_sr_dir, exist_ok=True)
            
            idx_label = idx if idx is not None else 0
            
            # Standard est vs ref comparison
            save_middle_slices(
                final_ct,
                f"final_{idx_label}",
                save_dir=final_sr_dir,
                ref_volume=ref_vol
            )
            
            # 3-way comparison: estimated | conditions (upsampled) | reference
            if cond_tensor is not None:
                save_sr_comparison(
                    estimated=final_ct,
                    conditions=cond_tensor,
                    ref_volume=ref_vol,
                    step=f"final_{idx_label}",
                    save_dir=final_sr_dir,
                )
            
            print(f"[+] Final SR volume saved to: {final_sr_dir}")

        return x0_preds, xs
