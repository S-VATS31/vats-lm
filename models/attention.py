from typing import Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast

from models.rope import RoPE
from models.kv_cache import KVCache
from gpu_setup import gpu_dtypes
from utils.attention_utils import extend_kv_heads, apply_qk_norm, check_contiguous

class CausalAttention(nn.Module):
    """Causal attention module.
    
    Args:
        d_model (int): Dimensionality of model embeddings.
        num_heads (int): Number of attention heads.
        query_groups (int): Number of query groups for GQA.
        rope_theta (float): Exponential base of RoPE.
        use_qkv_bias (bool): Whether to use QKV bias or not.
        use_o_bias (bool): Whether to use output bias or not.
        softmax_scale (float): Scaler for attention computation.
        use_windowed_attn (bool): Whether to use windowed attention.
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
        device: torch.device,
        dtype: torch.dtype
    ):
        super().__init__()
        
        assert d_model % num_heads == 0
        assert num_heads % query_groups == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.query_groups = query_groups
        self.softmax_scale = softmax_scale
        self.use_windowed_attn = use_windowed_attn
        self.device = device
        self.dtype = dtype
        self.head_dim = d_model // num_heads
        self.heads_per_group = num_heads // query_groups

        self.w_qkv = nn.Linear(
            d_model,
            num_heads*self.head_dim+2*query_groups*self.head_dim,
            bias=use_qkv_bias,
            device=device,
            dtype=dtype
        )
        self.w_o = nn.Linear(
            d_model,
            d_model,
            bias=use_o_bias,
            device=device,
            dtype=dtype
        )

        self.rope = RoPE(self.head_dim, rope_theta, device)

    def _setup_qkv(
        self,
        x: Tensor,
        use_qk_norm: bool = True,
        use_mqa: bool = False,
        qk_norm_eps: float = 1e-10,
        use_qk_rms_norm: bool = False,
        cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None,
        use_cache: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Set up QKV tensors.
        
        Args:
            x (Tensor): Input tensor of shape [B, T_tokens, d_model].
            use_qk_norm (bool): Whether use QK norm or not.
            use_mqa (bool): Whether to use MQA or not.
            use_qk_rms_norm (bool): Whether to use L2 or RMS QK norm.
            cache (Optional[KVCache]): KV caching module.
            layer_idx (Optional[int]): Layer of transformer stack.
            use_cache (bool): Whether to use KV caching or not.

        Returns:
            Tuple:
                - Tensor: Query tensor of shape [B, num_heads, T_tokens, head_dim].
                - Tensor: Key tensor of shape [B, num_heads, T_tokens, head_dim].
                - Tensor: Value tensor of shape [B, num_heads, T_tokens, head_dim].
        """
        B, T_tokens, _ = x.shape
        
        if T_tokens == 0:
            return (
                torch.empty(
                    B, self.num_heads, T_tokens, self.head_dim, 
                    device=x.device, dtype=x.dtype
                ),
                torch.empty(
                    B, self.num_heads, T_tokens, self.head_dim, 
                    device=x.device, dtype=x.dtype
                ),
                torch.empty(
                    B, self.num_heads, T_tokens, self.head_dim, 
                    device=x.device, dtype=x.dtype
                )
            )
        
        qkv = self.w_qkv(x)
        q, kv = torch.split(
            qkv, [self.num_heads*self.head_dim, 2*self.query_groups*self.head_dim], dim=-1
        )
        k, v = kv.chunk(2, dim=-1)

        q = q.view(B, T_tokens, self.num_heads, self.head_dim)
        k = k.view(B, T_tokens, self.query_groups, self.head_dim)
        v = v.view(B, T_tokens, self.query_groups, self.head_dim)

        if use_qk_norm:
            q = apply_qk_norm(q, qk_norm_eps, use_qk_rms_norm)
            k = apply_qk_norm(q, qk_norm_eps, use_qk_rms_norm)

        q = check_contiguous(self.rope(q))
        k = check_contiguous(self.rope(k))

        k = extend_kv_heads(k, dim=1, repeats=self.heads_per_group, use_mqa=use_mqa)
        v = extend_kv_heads(v, dim=1, repeats=self.heads_per_group, use_mqa=use_mqa)

        if use_cache and cache is not None and layer_idx is not None:
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            cache.update(k, v, layer_idx)
            past_k, past_v = cache.get(layer_idx)
            # Concatenate over sequence length dimension
            if past_k is not None and past_v is not None:
                k = torch.cat([k, past_k], dim=2)
                v = torch.cat([v, past_v], dim=2)
            return (
                check_contiguous(q.permute(0, 2, 1, 3)), 
                check_contiguous(k), 
                check_contiguous(v)
            )
        
        # b, h, t, d
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        return check_contiguous(q), check_contiguous(k), check_contiguous(v)

    def _optimized_attention(
        self,
        x: Tensor,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        use_causal: bool = True,
        padding_mask: Optional[Tensor] = None,
        left_window: int = -1,
        right_window: int = -1,
        softcap: float = 0.0,
    ) -> Tensor:
        """Optimized attention utilizing flash attention 3.
        
        Args:
            x (Tensor): Input tensor of shape [B, T_tokens, num_heads].
            q (Tensor): Query tensor.
            k (Tensor): Key tensor.
            v (Tensor): Value tensor.
            use_causal (bool): Whether to use causal masking or not.
            padding_mask (Optional[Tensor]): Padding tensor of shape [B, T_tokens].
            left_window (int): Left window for windowed attention.
                -1 means no bounds.
            right_window (int): Right window for windowed attention.
            softcap (float): Softcap value for attention.
                0.0 means no softcap.

        Returns:
            Tensor: Output tensor of shape [B, T_tokens, d_model].
        """
        if (
            x.device.type == "cuda" and
            q.device.type == "cuda" and
            k.device.type == "cuda" and
            v.device.type == "cuda" and
            x.dtype in gpu_dtypes and
            q.dtype in gpu_dtypes and
            k.dtype in gpu_dtypes and
            v.dtype in gpu_dtypes
        ):
            pass
        else:
            return self._dot_product_attention(
                x, q, k, v, use_causal, padding_mask
            )

    def _dot_product_attention(
        self,
        x: Tensor,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        use_causal: bool = True,
        padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Scaled dot product attention.
        
        Args:
            x (Tensor): Input tensor of shape [B, T_tokens, d_model].
            q (Tensor): Query tensor.
            k (Tensor): Key tensor.
            v (Tensor): Value tensor.
            use_causal (bool): Whether to use causal masking or not.
            padding_mask (Optional[Tensor]): Padding tensor of shape [B, T_tokens].

        Returns:
            Tensor: Output tensor of shape [B, T_tokens, d_model].
        """
        B, _, T_q, _= q.shape
        T_k = k.size(2)

        if q.numel() == 0 or k.numel() == 0 or v.numel() == 0:
            return torch.empty_like(x, device=x.device, dtype=x.dtype)
        
        attn_mask = None
        if padding_mask is not None:
            padding_mask = padding_mask.bool() # [B, T_q]
            query_padding_mask = padding_mask[:, None, :, None] # [B, 1, T_q, 1]
            if T_k > T_q:
                past_padding_mask = torch.ones(
                    B, T_k-T_q,
                    device=padding_mask.device,
                    dtype=torch.bool
                )
                full_key_padding_mask = torch.cat([padding_mask, past_padding_mask], dim=-1)
                key_padding_mask = full_key_padding_mask[:, None, None, :] # [B, 1, 1, T_k]
            else:
                key_padding_mask = padding_mask[:, None, None, :] # [B, 1, 1, T_k]
            attn_mask = torch.logical_or(query_padding_mask, key_padding_mask) # [B, 1, T_q, T_k]
            if use_causal:
                # causal_mask: [T_q, T_k]
                causal_mask = torch.triu(
                    torch.ones(T_q, T_k, device=attn_mask.device, dtype=torch.bool),
                    diagonal=1
                )
                causal_mask = causal_mask[None, None, :, :]
                attn_mask = torch.logical_or(attn_mask, causal_mask)
        else:
            if use_causal:
                # causal_mask: [T_q, T_k]
                causal_mask = torch.triu(
                    torch.ones(T_q, T_k, device=attn_mask.device, dtype=torch.bool),
                    diagonal=1
                )
                attn_mask = causal_mask[None, None, :, :]

        # out: [B, num_heads, T_tokens, head_dim]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = check_contiguous(out.transpose(1, 2))
        out = out.view(B, T_q, -1) # [B, T_q, d_model]

        return self.w_o(out)

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
        return_qkv: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
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
            return_qkv (bool): Whether to return QKV tensors or not.

        Returns:
            Union:
                - Tensor: Output tensor.
                - Tuple: Output tensor and QKV tensors.
        """
        with autocast(device_type=x.device.type):
            q, k, v = self._setup_qkv(
                x,
                use_qk_norm=use_qk_norm,
                use_mqa=use_mqa,
                qk_norm_eps=qk_norm_eps,
                use_qk_rms_norm=use_qk_rms_norm,
                cache=cache,
                layer_idx=layer_idx,
                use_cache=use_cache
            )
            out = self._optimized_attention(
                x, q, k, v,
                use_causal=use_causal,
                padding_mask=padding_mask,
                left_window=left_window,
                right_window=right_window,
                softcap=softcap,
            )
            if return_qkv:
                return out, q, k, v
            return out


class CausalAttentionBlock(nn.Module):
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
        device: torch.device,
        dtype: torch.dtype
    ):
        super().__init__()
        
        self.attn = CausalAttention(
            d_model=d_model,
            num_heads=num_heads,
            query_groups=query_groups,
            rope_theta=rope_theta,
            use_qkv_bias=use_qkv_bias,
            use_o_bias=use_o_bias,
            softmax_scale=softmax_scale,
            use_windowed_attn=use_windowed_attn,
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
            return x + self.dropout(self.attn(
                self.rms_norm(x),
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
