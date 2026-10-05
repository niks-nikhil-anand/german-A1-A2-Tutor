import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Converts token IDs into dense vectors.
    
    Example:
        token ID 42
        ->
        [0.12, -0.31, 0.44, ...]
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).

    Provides positional information to attention
    by rotating query and key vectors.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float = 10000.0
    ):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")

        # Frequencies used for the rotations.
        inv_freq = 1.0 / (
            theta ** (
                torch.arange(0, head_dim, 2).float() / head_dim
            )
        )

        positions = torch.arange(max_seq_len).float()

        # [max_seq_len, head_dim / 2]
        freqs = torch.outer(positions, inv_freq)

        # Store cosine and sine values.
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor
    ):
        """
        q and k shape:

        [batch, heads, sequence_length, head_dim]
        """

        seq_len = q.shape[-2]

        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]

        # Add dimensions for broadcasting:
        # [seq_len, head_dim/2]
        # ->
        # [1, 1, seq_len, head_dim/2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q = self._rotate(q, cos, sin)
        k = self._rotate(k, cos, sin)

        return q, k

    @staticmethod
    def _rotate(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> torch.Tensor:

        # Split even and odd dimensions.
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # Apply rotation.
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        # Interleave even and odd dimensions again.
        rotated = torch.stack(
            [rotated_even, rotated_odd],
            dim=-1
        )

        return rotated.flatten(-2)