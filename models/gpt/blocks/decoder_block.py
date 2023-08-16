from typing import Optional
import torch
from torch import nn
from models.gpt.layers.layer_norm import LayerNorm
from models.gpt.layers.causal_self_attention import CausalSelfAttention
from models.gpt.layers.feed_forward import FeedForward


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout_prob: Optional[float] = 0.1,
    ):
        """Constructor class for the decoder block of the GPT model

        Args:
            d_model (int): Dimension of the Model
            d_ff (int): Hidden dimension of the Feed Forward Layer
            num_heads (int): Number of heads in the multi-head attention
            dropout_prob (float): Dropout probability
        """
        super(DecoderBlock, self).__init__()
        self.attention = CausalSelfAttention(d_model, num_heads)
        self.layer_norm1 = LayerNorm(d_model)

        self.encoder_attention = CausalSelfAttention(d_model, num_heads)
        self.layer_norm2 = LayerNorm(d_model)

        self.feed_forward = FeedForward(
            d_model=d_model, d_ff=d_ff, dropout=dropout_prob
        )  # Specifying arguments here to avoid ambiguity
        # self.layer_norm3 = LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, trg: torch.Tensor, trg_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the decoder block using the pre-Norm Architecture from the original paper

        Args:
            trg (torch.Tensor): Target tensor of shape (batch_size, seq_len, embedding_dim)
            trg_mask (torch.Tensor): Target mask tensor of shape (batch_size, seq_len, seq_len)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embedding_dim)
        """
        # Layer Norm
        trg_norm = self.layer_norm1(trg)
        # Self attention
        self_attention, att_weights = self.attention(
            trg_norm, trg_norm, trg_norm, trg_mask
        )
        # Residual connection and dropout
        trg = trg + self.dropout(self_attention)
        # Layer normalization
        trg_norm = self.layer_norm2(trg)
        # Feed forward
        feed_forward = self.feed_forward(trg_norm)
        # Residual connection and dropout
        out = trg + self.dropout(feed_forward)

        return out
