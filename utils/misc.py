import torch.nn as nn

def count_params(model: nn.Module) -> int:
    """Count parameters of a model.
    
    Args:
        model (nn.Module): Model to have parameters counted.

    Returns:
        int: Total trainable parameters in model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
