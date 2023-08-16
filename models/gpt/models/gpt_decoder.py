import math

import torch
from torch import nn
from models.gpt.blocks.decoder_block import DecoderBlock
from models.gpt.embeddings.token_positional import TransformerEmbeddings
from models.gpt.layers.layer_norm import LayerNorm


class GPTDecoder(nn.Module):
    def __init__(
        self,
        vocab_size_dec: int,
        d_model: int,
        max_seq_len: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        dropout_prob: float,
    ):
        """Constructor class for the decoder of GPT model which is a decoder-only transformer

        Args:
            vocab_size_dec (int): Size of the vocabulary of the decoder
            d_model (int): Dimension of the model
            max_seq_len (int): Maximum sequence length
            num_layers (int): Number of decoder layers
            num_heads (int): Number of heads in the multi-head attention
            d_ff (int): Hidden dimension of the Feed Forward Layer
            dropout_prob (float): Dropout probability
        """
        super(GPTDecoder, self).__init__()

        self.embedding_dim = d_model
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers

        self.embedding = TransformerEmbeddings(
            vocab_size=vocab_size_dec,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout_prob,
        )

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_heads=num_heads,
                    dropout_prob=dropout_prob,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = LayerNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size_dec)

        # Tie the weights of the embedding and linear layer used in GPT
        self.lm_head.weight = self.embedding.token_embeddings.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize the weights of the model using normal distribution with mean=0 and std=0.02 and zero out the bias
        of the linear layer
        Args:
            module (nn.Module): Module of the model
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(
                mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers)
            )
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, trg: torch.Tensor, trg_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GPT decoder

        Args:
            trg (torch.Tensor): Target tensor of shape (batch_size, seq_len)
            trg_mask (torch.Tensor): Target mask tensor of shape (batch_size, seq_len, seq_len)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, vocab_size) - logits
        """
        # Apply token and positional embeddings
        trg = self.embedding(trg)

        # Apply decoder blocks
        for decoder_block in self.decoder_blocks:
            trg = decoder_block(trg, trg_mask)

        # Apply final layer norm
        trg = self.final_norm(trg)

        # Apply linear layer
        output = self.lm_head(trg)
        return output
