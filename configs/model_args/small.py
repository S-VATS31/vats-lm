
import math
import torch
from dataclasses import dataclass

@dataclass
class ModelArgs:
    """Extra small configuration of model arguments."""
    d_model: int = 256
    num_heads: int = 16
    query_groups: int = 2
    d_ffn: int = 1024
    num_layers: int = 8
    dropout_prob: float = 0.1
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-7
    qk_norm_eps: float = 1e-8
    left_window: int = 128
    right_window: int = 0
    vocab_size: int = 512
    max_seq_len: int = 128
    gradient_checkpointing: bool = True
    use_qkv_bias: bool = False
    use_o_bias: bool = False
    use_mqa: bool = False
    softmax_scale: float = math.sqrt(256//16)
    use_mlp_bias: bool = False
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32
    use_qk_norm: bool = True
    use_qk_rms_norm: bool = False
    use_weight_tying: bool = True
    use_causal: bool = True
    use_windowed_attn: bool = True
    softcap: float = 20.0
