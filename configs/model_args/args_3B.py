import math
import torch
from dataclasses import dataclass

@dataclass
class ModelArgs:
    """Extra small configuration of model arguments."""
    d_model: int = 2560
    num_heads: int = 64
    query_groups: int = 16
    d_ffn: int = 10240
    num_layers: int = 31
    dropout_prob: float = 0.2
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-7
    qk_norm_eps: float = 1e-8
    left_window: int = -1
    right_window: int = 0
    vocab_size: int = 32768
    max_seq_len: int = 512
    gradient_checkpointing: bool = True
    use_qkv_bias: bool = False
    use_o_bias: bool = False
    use_mqa: bool = False
    softmax_scale: float = 1/ math.sqrt(2560//32)
    use_mlp_bias: bool = False
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32
    use_qk_norm: bool = True
    use_qk_rms_norm: bool = False
    use_weight_tying: bool = True
    use_causal: bool = True
    use_windowed_attn: bool = True
    softcap: float = 20.0
