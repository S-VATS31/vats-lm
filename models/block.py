import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast

from models.attention import CausalAttentionBlock

class CausalTransformerBlock(nn.Module):
    def __init__(
        d_model: int,
        num_heads: int,
        query_groups: int,
        rope_theta: float,
        use_qkv_bias: bool,
        use_o_bias: bool,
        softmax_scale: bool,
        use_windowed_attn: bool,
        rms_norm_eps: float,
        dropout_prob: float,
        d_ffn: int,
        use_mlp_bias: bool,
        device: torch.device,
        dtype: torch.dtype
    ):
        super().__init__()

