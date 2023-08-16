from typing import Optional
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(
        self, d_model: int, d_ff: Optional[int] = None, dropout: Optional[float] = 0.1
    ):
        """Constructor class for the Position-wise Feed Forward layer for GPT implementation

        Args:
            d_model (int): Dimension of the model
            d_ff (int): Dimension of the feed forward layer - if None, will default to 4 * d_model
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(FeedForward, self).__init__()

        if d_ff is None:
            d_ff = 4 * d_model  # This is the default setting in the GPT paper

        # Create the feed forward layers
        self.linear_1 = nn.Linear(d_model, d_ff)
        # For GPT, we use GeLU activation instead of ReLU
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Position-wise Feed Forward layer for GPT implementation

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x
