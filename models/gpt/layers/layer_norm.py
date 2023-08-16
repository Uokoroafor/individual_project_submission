from typing import Optional
import torch
from torch import nn as nn
import torch.nn.functional as F


class LayerNorm(nn.LayerNorm):
    """Layer Normalization class"""

    def __init__(self, d_model: int, eps: Optional[float] = 1e-6):
        """Constructor class for LayerNorm

        Args:
            d_model (int): Dimension of the model
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-6.
        """
        super().__init__(d_model, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the layer normalization layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        return super().forward(x)
