from __future__ import annotations

from models.models import DiT

# Initialize the DiT_models dictionary
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
    # Define hidden sizes for different model scales
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
    def model_fn(**kwargs):
        return create_dit_model(**model_args, **kwargs)
    
    # Add to DiT_models dictionary
    DiT_models[name] = model_fn
    return model_fn

def load_model(config):
    # Check if the model name is in the DiT models dictionary
    if config.model.name not in DiT_models:
        raise ValueError(f"Model name {config.model.name} is not recognized.")
    
    # Check if the model name is in the BiXT models dictionary
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

# Register XS models with patch size 12
register_dit_model("DiT-XS/12/0", size="XS", patch_size=12, depth=0, stride=8, 
                   padding=2, num_heads=6, mlp_ratio=4.0)
register_dit_model("DiT-XS/12/1", size="XS", patch_size=12, depth=1, stride=8, 
                   padding=2, num_heads=6, mlp_ratio=4.0)
register_dit_model("DiT-XS/12/2", size="XS", patch_size=12, depth=2, stride=8, 
                   padding=2, num_heads=6, mlp_ratio=4.0)

# Popular patch size 4 models for high-resolution tasks
register_dit_model("DiT-XS/4/0", size="XS", patch_size=4, depth=0, stride=4, padding=0,
                   num_heads=6, mlp_ratio=4.0)
register_dit_model("DiT-XS/4/1", size="XS", patch_size=4, depth=1, stride=2, padding=0,
                   num_heads=6, mlp_ratio=4.0)

