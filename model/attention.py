import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import RotaryEmbedding


class MultiHeadSelfAttention(nn.Module):
    """
    Causal multi-head self-attention.

    Input:
        [batch, sequence_length, d_model]

    Output:
        [batch, sequence_length, d_model]
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads."
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        # Create Q, K and V in one operation.
        self.qkv_proj = nn.Linear(
            d_model,
            3 * d_model,
            bias=False,
        )

        # Final projection after attention heads are combined.
        self.out_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        # RoPE is applied to Q and K.
        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
            [B, T, C]

        B = batch size
        T = sequence length
        C = d_model
        """

        B, T, C = x.shape

        # --------------------------------
        # 1. Create Query, Key and Value
        # --------------------------------

        qkv = self.qkv_proj(x)

        # [B, T, 3*C] -> three tensors [B, T, C]
        q, k, v = qkv.chunk(3, dim=-1)

        # --------------------------------
        # 2. Split into attention heads
        # --------------------------------

        # [B, T, C]
        # ->
        # [B, T, heads, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)

        # [B, T, heads, head_dim]
        # ->
        # [B, heads, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # --------------------------------
        # 3. Apply RoPE
        # --------------------------------

        q, k = self.rope(q, k)

        # --------------------------------
        # 4. Causal self-attention
        # --------------------------------

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # --------------------------------
        # 5. Combine attention heads
        # --------------------------------

        # [B, heads, T, head_dim]
        # ->
        # [B, T, heads, head_dim]
        y = y.transpose(1, 2)

        # ->
        # [B, T, d_model]
        y = y.contiguous().view(B, T, C)

        # --------------------------------
        # 6. Output projection
        # --------------------------------

        y = self.out_proj(y)

        return y