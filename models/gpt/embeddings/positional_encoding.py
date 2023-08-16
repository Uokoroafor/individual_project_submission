import torch
from torch import nn


class PositionalEncoding(nn.Embedding):
    def __init__(self, d_model: int, max_len: int):
        """
        Class for positional encoding in GPT. This is a fixed positional encoding which is learned during training.
        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the positional encoding layer. Here encodings are learned during training.
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Positional encodings of shape (batch_size, seq_len, d_model)
        """
        b, t, e = x.size()  # b: batch_size, t: seq_len, e: d_model
        positions = (
            torch.arange(t, device=x.device, dtype=x.dtype).expand(b, t).contiguous()
        )  # (batch_size, seq_len)
        positions = positions.long()  # (batch_size, seq_len)
        encodings = super(PositionalEncoding, self).forward(
            positions
        )  # (batch_size, seq_len, d_model)
        return x + encodings  # (batch_size, seq_len, d_model)
