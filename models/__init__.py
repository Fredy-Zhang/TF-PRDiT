from __future__ import annotations

from models.models import DiT

DiT_models = {}

def create_dit_model(
    size="S", 
    patch_size=12, 
    depth=0, 
    stride=8, 
    padding=2, 
    num_heads=6, 
    mlp_ratio=4.0, 
    **kwargs
):
    """Create a configured 3D DiT instance.

    Args:
        size: Model scale key, one of ``XS``, ``S``, ``B``, ``L``, or ``XL``.
        patch_size: Cubic patch size used by the patch extractor.
        depth: Number of transformer refinement blocks.
        stride: Patch extraction stride.
        padding: Reflect padding before patch extraction.
        num_heads: Number of transformer attention heads.
        mlp_ratio: MLP expansion ratio in transformer blocks.
        **kwargs: Additional ``DiT`` constructor arguments.
    """
    hidden_sizes = {
        "XS": 192,
        "S": 384,
        "B": 768,
        "L": 1024,
        "XL": 1152
    }
    
    if size not in hidden_sizes:
        raise ValueError(f"Invalid model size: {size}. Must be one of {list(hidden_sizes.keys())}")
    
    hidden_size = hidden_sizes[size]
    
    return DiT(
        depth=depth,
        hidden_size=hidden_size,
        patch_size=patch_size,
        stride=stride,
        padding=padding,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        **kwargs
    )

def register_dit_model(name, **model_args):
    """Register a named DiT factory in ``DiT_models``."""
    def model_fn(**kwargs):
        return create_dit_model(**model_args, **kwargs)
    
    DiT_models[name] = model_fn
    return model_fn

def load_model(config):
    """Instantiate the model described by a loaded YAML config."""
    if config.model.name not in DiT_models:
        raise ValueError(f"Model name {config.model.name} is not recognized.")
    
    return DiT_models[config.model.name](
        input_size=config.data.image_size,
        in_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        learn_sigma=True if config.model.out_channels == 2 else False,
        flash_attn=config.model.flash_attn
    )

#################################################################################
#                        Register Standard Model Variants                        #
#################################################################################
def register_all_dit_models(
    patch_sizes: list[int] = (10, 12, 14, 16),
    stride: int = 8,
    sizes_num_heads: list[tuple[str,int]] = (("XS", 6), ("S", 6), ("B", 12), ("L", 16), ("XL", 16)),
    depths: list[int] = (0, 1, 2, 3, 4, 8, 10, 12),
    mlp_ratio: float = 4.0,
):
    """Register the DiT model names supported by the config files."""
    for P in patch_sizes:
        padding = (P - stride) // 2
        for size, num_heads in sizes_num_heads:
            for depth in depths:
                register_dit_model(
                    name=f"DiT-{size}/{P}/{depth}",
                    size=size,
                    patch_size=P,
                    depth=depth,
                    stride=stride,
                    padding=padding,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )

register_all_dit_models()
