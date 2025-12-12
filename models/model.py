from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor, LongTensor
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint

from configs.model_args.args_7B import ModelArgs
from models.block import CausalTransformerBlock
from models.kv_cache import KVCache

class CausalTransformer(nn.Module):
    """Causal text transformer.
    
    Args:
        model_args (ModelArgs): Model hyperparameters.
    """
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        
        self.model_args = model_args
        
        self.embedding = nn.Embedding(
            model_args.vocab_size,
            model_args.d_model,
            device=model_args.device,
            dtype=model_args.dtype
        )

        self.layers = nn.ModuleList([
            CausalTransformerBlock(
                d_model=model_args.d_model,
                num_heads=model_args.num_heads,
                query_groups=model_args.query_groups,
                rope_theta=model_args.rope_theta,
                use_qkv_bias=model_args.use_qkv_bias,
                use_o_bias=model_args.use_o_bias,
                softmax_scale=model_args.softmax_scale,
                use_windowed_attn=model_args.use_windowed_attn,
                rms_norm_eps=model_args.rms_norm_eps,
                dropout_prob=model_args.dropout_prob,
                d_ffn=model_args.d_ffn,
                use_mlp_bias=model_args.use_mlp_bias,
                device=model_args.device,
                dtype=model_args.dtype
            ) for _ in range(model_args.num_layers)
        ])

        self.kv_cache = KVCache(
            num_heads=model_args.num_heads,
            head_dim=model_args.d_model//model_args.num_heads,
            num_layers=model_args.num_layers,
            max_seq_len=model_args.max_seq_len,
            device=model_args.device,
            dtype=model_args.dtype
        )

        self.rms_norm = nn.RMSNorm(
            model_args.d_model,
            model_args.rms_norm_eps,
            device=model_args.device,
            dtype=model_args.dtype
        )

        self.dropout = nn.Dropout(p=model_args.dropout_prob)

        self.lm_head = nn.Linear(
            model_args.d_model,
            model_args.vocab_size,
            bias=False,
            device=model_args.device,
            dtype=model_args.dtype
        )

        if model_args.use_weight_tying:
            self.lm_head.weight = self.embedding.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for all PyTorch modules.
        
        Args:
            module (nn.Module): Module to be initialized.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: LongTensor,
        use_cache: bool = False,
        padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Run forward pass.
        
        Args:
            input_ids (LongTensor): Input tensor of shape [B, T_tokens].
            use_cache (bool): Whether to use KV caching or not.
            padding_mask (Optional[Tensor]): Padding tensor of shape [B, T_tokens].
        """
        with autocast(device_type=input_ids.device.type):
            if input_ids.dtype != torch.int64:
                input_ids = input_ids.to(torch.int64)
            x = self.dropout(self.embedding(input_ids)) # [B, T_tokens, d_model]

            # loop through transformer layers
            for layer_idx, layer in enumerate(self.layers):
                if self.model_args.gradient_checkpointing:
                    x = checkpoint(
                        layer,
                        x,
                        self.model_args.use_qk_norm,
                        self.model_args.use_mqa,
                        self.model_args.qk_norm_eps,
                        self.model_args.use_qk_rms_norm,
                        self.kv_cache,
                        layer_idx,
                        use_cache,
                        self.model_args.use_causal,
                        self.model_args.left_window,
                        self.model_args.right_window,
                        self.model_args.right_window,
                        self.model_args.softcap,
                        padding_mask,
                        use_reentrant=False
                    )
                else:
                    x = layer(
                        x,
                        use_qk_norm=self.model_args.use_qk_norm,
                        use_mqa=self.model_args.use_mqa,
                        qk_norm_eps=self.model_args.qk_norm_eps,
                        use_qk_rms_norm=self.model_args.use_qk_rms_norm,
                        cache=self.kv_cache,
                        layer_idx=layer_idx,
                        use_cache=use_cache,
                        use_causal=self.model_args.use_causal,
                        left_window=self.model_args.left_window,
                        right_window=self.model_args.right_window,
                        softcap=self.model_args.softcap,
                        padding_mask=padding_mask
                    )

            # final RMSNorm
            x = self.rms_norm(x)

            # get logits
            logits = self.lm_head(x) # [B, T, vocab_size]

            return logits

from utils.misc import count_params
params = count_params(CausalTransformer(ModelArgs()))
print(f"{params:,}")