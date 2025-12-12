from typing import Tuple

import torch
from torch import Tensor

class KVCache:
    """KV caching module for attention.
    
    Args:
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        num_layers (int): Number of transformer layers.
        max_seq_len (int): Maximum sequence length to allocate KV cache.
        device (torch.device): Device of key and value tensors.
        dtype (torch.dtype): Data type of key and value tensors.
    """
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_layers: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        self.cache = None
        self.batch_size = None
        self.current_seq_len = None

    def _initialize(self, batch_size: int) -> None:
        """Initialize cache using batch size of key and value tensors.
        
        Args:
            batch_size (int): Batch size of key and value tensors.
        """
        self.batch_size = batch_size
        self.current_seq_len = 0

        self.cache = [
            {
                "k": torch.ones((
                    batch_size, self.num_heads, self.max_seq_len, self.head_dim
                ), device=self.device, dtype=self.dtype),
                "v": torch.ones((
                    batch_size, self.num_heads, self.max_seq_len, self.head_dim
                ), device=self.device, dtype=self.dtype)
            }
            for _ in range(self.num_layers)
        ]

    def update(self, k: Tensor, v: Tensor, layer_idx: int) -> None:
        """Update KV cache using key and value tensors.
        
        Args:
            k (Tensor): Input key tensor of shape [B, num_heads, T_tokens, head_dim].
            v (Tensor): Input value tensor of shape [B, num_heads, T_tokens, head_dim].
            layer_idx (int): Layer to update KV's with respect to.
        """
        if self.cache is None or self.batch_size is None:
            self._initialize(k.size(0))

        # Get new tokens
        new_tokens = k.size(2)

        # Cache full
        if self.current_seq_len + new_tokens >= self.max_seq_len:
            current_space = self.max_seq_len - self.current_seq_len
            if current_space <= 0:
                return
            
            # Truncate over sequence length dim
            k = k[:, :, :current_space]
            v = v[:, :, :current_space]
            new_tokens = current_space
        
        # Update cache over sequence length dim
        self.cache["k"][layer_idx][:, :, self.current_seq_len:self.current_seq_len+new_tokens] = k
        self.cache["v"][layer_idx][:, :, self.current_seq_len:self.current_seq_len+new_tokens] = v

        self.current_seq_len += new_tokens

    def get(self, layer_idx: int) -> Tuple[Tensor, Tensor]:
        """Get key and value tensors up to all tokens.
        
        Args:
            layer_idx (int): Layer of key and value tensors of cache.
        
        Returns:
            Tuple:
                - Tensor: Key tensor.
                - Tensor: Value tensor.
        """
        if self.cache is None:
            return None, None
        
        return (
            self.cache["k"][layer_idx][:, :, :self.current_seq_len],
            self.cache["v"][layer_idx][:, :, :self.current_seq_len]
        )
    
    def reset(self) -> None:
        """Reset all states to None."""
        self.batch_size = None
        self.current_seq_len = None
        self.cache = None
