from torch.nn import modules
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadSelfAttention
from .mlp import SwiGLU
from .normalization import RMSNorm
from .embeddings import TokenEmbedding


class TransformerBlock(nn.Module):
    """
    One Transformer block.

    Structure:

        x
        │
        ├── RMSNorm
        │
        ├── Attention
        │
        └── + residual
        │
        ├── RMSNorm
        │
        ├── SwiGLU
        │
        └── + residual
    """

    def __init__(self, config):
        super().__init__()

        self.attention_norm = RMSNorm(
            config.d_model,
            config.rms_norm_eps,
        )

        self.attention = MultiHeadSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            max_seq_len=config.context_length,
            rope_theta=config.rope_theta,
            dropout=config.dropout,
        )

        self.ffn_norm = RMSNorm(
            config.d_model,
            config.rms_norm_eps,
        )

        self.ffn = SwiGLU(
            d_model=config.d_model,
            d_ff=config.d_ff,
        )

    def forward(self, x):

        # Attention + residual connection
        x = x + self.attention(
            self.attention_norm(x)
        )

        # Feed-forward network + residual connection
        x = x + self.ffn(
            self.ffn_norm(x)
        )

        return x


class TransformerLM(nn.Module):
    """
    Complete decoder-only language model.

    Input:
        token IDs

    Output:
        logits for every possible next token
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # Convert token IDs into vectors
        self.token_embedding = TokenEmbedding(
            config.vocab_size,
            config.d_model,
        )

        # Transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_layers)
            ]
        )

        # Final normalization
        self.final_norm = RMSNorm(
            config.d_model,
            config.rms_norm_eps,
        )

        # Convert hidden representation into vocabulary scores
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        self._init_weights()

    def _init_weights(self):

        # Standard initialization
        for module in self.modules():

            if isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02,
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02,
                )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        """
        input_ids:

            [batch, sequence_length]

        labels:

            [batch, sequence_length]

        Returns:

            logits:
                [batch, sequence_length, vocab_size]

            loss:
                scalar, if labels are provided
        """

        batch_size, seq_len = input_ids.shape

        if seq_len > self.config.context_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"context length {self.config.context_length}"
            )

        # ---------------------------------
        # 1. Token embedding
        # ---------------------------------

        x = self.token_embedding(input_ids)

        # ---------------------------------
        # 2. Transformer layers
        # ---------------------------------

        for layer in self.layers:
            x = layer(x)

        # ---------------------------------
        # 3. Final normalization
        # ---------------------------------

        x = self.final_norm(x)

        # ---------------------------------
        # 4. Vocabulary logits
        # ---------------------------------

        logits = self.lm_head(x)

        loss = None

        # ---------------------------------
        # 5. Next-token prediction loss
        # ---------------------------------

        if labels is not None:

            # Predict token t+1 from tokens <= t
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        return logits, loss


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )