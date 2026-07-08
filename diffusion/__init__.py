"""Factory for constructing the diffusion process used by training.

The cleaned paper code supports only the Image-and-Noise (IaN) objective:
the DiT model predicts both reconstructed noise and reconstructed image.
All current training configs therefore set ``model.out_channels`` to ``2``.
"""

from diffusion.image_noise_diffusion import IaNDiffusion


def loading_diffusion(config, rank=0):
    """Create the Image-and-Noise diffusion object used by ``train.py``.

    Args:
        config: Loaded training config. Requires ``config.model.out_channels == 2``.
        rank: Distributed rank. Rank 0 prints the selected diffusion type.

    Returns:
        An ``IaNDiffusion`` instance configured for 250 training timesteps.

    Raises:
        ValueError: If the configured model output is not the IaN two-channel
            output used by this project.
    """
    if config.model.out_channels != 2:
        raise ValueError(f"Unsupported out_channels: {config.model.out_channels}")
    if rank == 0:
        print("Loading the Image-and-Noise diffusion model")
    return IaNDiffusion(timestep_respacing="250", loss_type="l2", config=config)
