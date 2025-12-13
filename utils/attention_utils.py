import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

def extend_kv_heads(
    tensor: Tensor,
    dim: int,
    repeats: int,
    use_mqa: bool = False
) -> Tensor:
    """Extend KV heads for key and value tensors.
    
    Args:
        input (Tensor): Input key or value tensor.
        dim (int): Dimension to be repeated over.
        repeats (int): Number of repeats over certain dimension.
        use_mqa (bool): Whether to use MQA or not.

    Returns:
        Tensor: Key or value tensor with extended dimension.
    """
    if tensor.size(dim) == 1 and use_mqa:
        return tensor
    return tensor.repeat_interleave(repeats, dim=dim)

def apply_qk_norm(
    tensor: Tensor,
    eps: float = 1e-10,
    use_rms_norm: bool = False
) -> Tensor:
    """Apply QK normalization to query and key tensors.
    
    Args:
        tensor (Tensor): Input query or key tensor.
        eps (float): Epsilon value to avoid division by zero errors.
        use_rms_norm (bool): Whether to use RMS normalization or L2 normalization.
    """
    def _norm(y: Tensor, eps: float = 1e-10):
        norm = nn.RMSNorm(y.size(-1), eps=eps)
        return norm(y)
    if use_rms_norm:
        return _norm(tensor, eps=eps)
    return F.normalize(tensor, p=2, dim=-1, eps=eps)
    
def check_contiguous(tensor: Tensor) -> Tensor:
    """Check if a tensor is contiguous or not
    
    Args:
        tensor (Tensor): Input tensor to be checked.

    Returns:
        Tensor: Returns contiguous tensor if not already contiguous.
    """
    return tensor if tensor.is_contiguous() else tensor.contiguous()
