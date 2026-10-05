import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps

        # Learnable scale parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate root mean square
        rms = torch.rsqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )

        # Normalize and scale
        return x * rms * self.weight