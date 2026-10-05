import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Input:
        [batch, sequence_length, d_model]

    Output:
        [batch, sequence_length, d_model]
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
    ):
        super().__init__()

        # Gate branch
        self.gate_proj = nn.Linear(
            d_model,
            d_ff,
            bias=False,
        )

        # Information branch
        self.up_proj = nn.Linear(
            d_model,
            d_ff,
            bias=False,
        )

        # Project back to model dimension
        self.down_proj = nn.Linear(
            d_ff,
            d_model,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Gate branch
        gate = F.silu(self.gate_proj(x))

        # Information branch
        up = self.up_proj(x)

        # Combine the two branches
        hidden = gate * up

        # Return to d_model dimension
        return self.down_proj(hidden)