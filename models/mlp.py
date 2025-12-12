import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast

from gpu_setup import gpu_dtypes

class MLP(nn.Module):
    """MLP module.
    
    Args:
        d_model (int): Dimensionality of model embeddings.
        d_ffn (int): Dimensionality of feedforward network.
        dropout_prob (float): Dropout probability.
        use_mlp_bias (bool): Whether to use biases in SwiGLU computation.
        device (torch.device): Accelerator at use.
        dtype (torch.dtype): Data type of model parameters.
    """
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        dropout_prob: float,
        use_mlp_bias: bool,
        device: torch.device,
        dtype: torch.dtype
    ):
        super().__init__()
        
        self.weight1 = nn.Linear(
            d_model,
            d_ffn,
            bias=use_mlp_bias,
            device=device,
            dtype=dtype
        )
        self.weight2 = nn.Linear(
            d_ffn,
            d_model,
            bias=use_mlp_bias,
            device=device,
            dtype=dtype
        )
        self.weight3 = nn.Linear(
            d_model,
            d_ffn,
            bias=use_mlp_bias,
            device=device,
            dtype=dtype
        )
        self.dropout = nn.Dropout(p=dropout_prob)

    def _optimized_swiglu(self, x: Tensor) -> Tensor:
        """Optimized SwiGLU utilizing GPU implementation.
    
        Args:
            x (Tensor): Input tensor of shape [B, T_tokens, d_model].

        Returns:
            Tensor: Output tensor with same shape as input.
        """
        if x.device == "cuda" and x.dtype in gpu_dtypes:
            pass
        else:
            return self._swiglu(x)

    def _swiglu(self, x: Tensor) -> Tensor:
        """SwiGLU PyTorch implementation.
        
        Args:
            x (Tensor): Input tensor of shape [B, T_tokens, d_model].

        Returns:
            Tensor: Output tensor with same shape as input.
        """
        return self.dropout(
            self.weight2(F.silu(self.weight1(x)) * self.weight3(x))
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass
        
        Args:
            x (Tensor): Input tensor of shape [B, T_tokens, d_model].

        Returns:
            Tensor: Output tensor with same shape as input.
        """
        with autocast(device_type=x.device.type):
            return self._optimized_swiglu(x)


class MLPBlock(nn.Module):
    """MLP block.
    
    Args:
        d_model (int): Dimensionality of model embeddings.
        d_ffn (int): Dimensionality of feedforward network.
        dropout_prob (float): Dropout probability.
        use_mlp_bias (bool): Whether to use biases in SwiGLU computation.
        rms_norm_eps (float): Epsilon value for RMSNorm.
        device (torch.device): Accelerator at use.
        dtype (torch.dtype): Data type of model parameters.
    """
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        dropout_prob: float,
        use_mlp_bias: bool,
        rms_norm_eps: float,
        device: torch.device,
        dtype: torch.dtype
    ):
        super().__init__()

        self.mlp = MLP(
            d_model=d_model,
            d_ffn=d_ffn,
            dropout_prob=dropout_prob,
            use_mlp_bias=use_mlp_bias,
            device=device,
            dtype=dtype
        )
        self.rms_norm = nn.RMSNorm(
            d_model,
            eps=rms_norm_eps,
            device=device,
            dtype=dtype
        )
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.
        
        Args:
            x (Tensor): Input tensor of shape [B, T_tokens, d_model].

        Returns:
            Tensor: Output tensor with same shape as input.
        """
        with autocast(device_type=x.device.type):
            return self.dropout(self.mlp(self.rms_norm(x)))
        