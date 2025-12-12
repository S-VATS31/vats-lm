import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast

from gpu_setup import gpu_dtypes

class MLP(nn.Module):
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
        if x.device == "cuda" and x.dtype in gpu_dtypes:
            pass
        else:
            return self._swiglu(x)

    def _swiglu(self, x: Tensor) -> Tensor:
        return self.dropout(
            self.weight2(F.silu(self.weight1(x)) * self.weight3(x))
        )
    
    def forward(self, x: Tensor) -> Tensor:
        with autocast(device_type=x.device.type):
            return self._optimized_swiglu(x)
