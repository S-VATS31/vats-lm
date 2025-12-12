from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast

from models.mlp import MLPBlock
from models.kv_cache import KVCache
from models.attention import CausalAttentionBlock

class CausalTransformerBlock(nn.Module):
    """Causal attention block.
    
    Args:
        d_model (int): Dimensionality of model embeddings.
        num_heads (int): Number of attention heads.
        query_groups (int): Number of query groups for GQA.
        rope_theta (float): Exponential base of RoPE.
        use_qkv_bias (bool): Whether to use QKV bias or not.
        use_o_bias (bool): Whether to use output bias or not.
        softmax_scale (float): Scaler for attention computation.
        use_windowed_attn (bool): Whether to use windowed attention.
        rms_norm_eps (float): Epsilon value for RMSNorm.
        dropout_prob (float): Dropout probability.
        d_ffn (int): Dimensionality of feedforward network.
        use_mlp_bias (bool): Whether to use biases in SwiGLU computation.
        device (torch.device): Accelerator at use.
        dtype (torch.dtype): Data type of model parameters.
    """
    def __init__(
        self,
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

        self.attn_block = CausalAttentionBlock(
            d_model=d_model,
            num_heads=num_heads,
            query_groups=query_groups,
            rope_theta=rope_theta,
            use_qkv_bias=use_qkv_bias,
            use_o_bias=use_o_bias,
            softmax_scale=softmax_scale,
            use_windowed_attn=use_windowed_attn,
            rms_norm_eps=rms_norm_eps,
            dropout_prob=dropout_prob,
            device=device,
            dtype=dtype
        )
        self.mlp_block = MLPBlock(
            d_model=d_model,
            d_ffn=d_ffn,
            dropout_prob=dropout_prob,
            use_mlp_bias=use_mlp_bias,
            rms_norm_eps=rms_norm_eps,
            device=device,
            dtype=dtype
        )
    def forward(
        self,
        x: Tensor,
        use_qk_norm: bool = True,
        use_mqa: bool = False,
        qk_norm_eps: float = 1e-10,
        use_qk_rms_norm: bool = False,
        cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None,
        use_cache: bool = False,
        use_causal: bool = False,
        left_window: int = -1,
        right_window: int = -1,
        softcap: float = 0.0,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Run forward pass.
        
        Args:
            x (Tensor): Input tensor.
            us_qk_norm (bool): Whether to use QK norm or not.
            use_mqa (bool): Whether to use MQA or not.
            qk_norm_eps (float): Epsilon value for QK norm.
            use_qk_rms_norm (bool): Whether to use L2 or RMS QK norm.
            layer_idx (Optional[int]): Current layer of transformer stack.
            use_cache (bool): Whether to use KV caching or not.
            use_causal (bool): Whether to use causal masking or not.
            left_window (int): Left window for windowed attention.
                -1 means no bounds.
            right_window (int): Right window for windowed attention.
                -1 means no bounds.
            softcap (float): Softcap value for attention.
                0.0 means no softcap.
            padding_mask (Optional[Tensor]): Padding tensor of shape [B, T_tokens].

        Returns:
            Tensor: Output tensor with same shape as input.
        """
        with autocast(device_type=x.device.type):
            return self.mlp_block(self.attn_block(
                x,
                use_qk_norm=use_qk_norm,
                use_mqa=use_mqa,
                qk_norm_eps=qk_norm_eps,
                use_qk_rms_norm=use_qk_rms_norm,
                cache=cache,
                layer_idx=layer_idx,
                use_cache=use_cache,
                use_causal=use_causal,
                left_window=left_window,
                right_window=right_window,
                softcap=softcap,
                padding_mask=padding_mask
            ))
        