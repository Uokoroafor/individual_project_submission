from typing import Optional
import torch
from torch import nn
from models.gpt.layers.layer_norm import LayerNorm
from models.gpt.layers.self_attention import SelfAttention
from models.gpt.layers.feed_forward import FeedForward


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout_prob: Optional[float] = 0.1,
    ):
        """Constructor class for the Encoder Block of the GPT model - Note GPT is a decoder only model so this is inferred from the original paper

        Args:
            d_model (int): Dimension of the Model
            d_ff (int): Hidden dimension of the Feed Forward Layer
            num_heads (int): Number of heads in the multi-head attention
            dropout_prob (float): Dropout probability
        """
        super(EncoderBlock, self).__init__()
        self.attention = SelfAttention(d_model, num_heads)
        self.layer_norm1 = LayerNorm(d_model)

        self.encoder_attention = SelfAttention(d_model, num_heads)
        self.layer_norm2 = LayerNorm(d_model)

        self.feed_forward = FeedForward(
            d_model=d_model, d_ff=d_ff, dropout=dropout_prob
        )  # Specifying arguments here to avoid ambiguity

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the encoder block also using pre-Norm Architecture.

        Args:
            src (torch.Tensor): Source tensor of shape (batch_size, seq_len, embedding_dim)
            src_mask (torch.Tensor): Source mask tensor of shape (batch_size, seq_len, seq_len)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embedding_dim)
        """
        # Layer Norm
        src_norm = self.layer_norm1(src)
        # Self attention
        self_attention, _ = self.attention(src_norm, src_norm, src_norm, src_mask)
        # Residual connection and dropout
        src = src + self.dropout(self_attention)
        # Layer normalization
        src_norm = self.layer_norm2(src)
        # Feed forward
        feed_forward = self.feed_forward(src_norm)
        # Residual connection and dropout
        out = src + self.dropout(feed_forward)

        return out
