import math
from typing import Optional
import torch
from torch import nn
from models.gpt.blocks.encoder_block import EncoderBlock
from models.gpt.embeddings.token_positional import TransformerEmbeddings
from models.gpt.layers.layer_norm import LayerNorm


class GPTEncoder(nn.Module):
    def __init__(
        self,
        vocab_size_enc: int,
        output_size: int,
        d_model: int,
        max_seq_len: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        dropout_prob: float,
        pooling: Optional[str] = "mean",
    ):
        """Constructor class for the encoder variant of a GPT model which is a decoder-only transformer.

        Args:
            vocab_size_enc (int): Size of the vocabulary of the encoder
            output_size (int): Size of the vocabulary of the decoder
            d_model (int): Dimension of the model
            max_seq_len (int): Maximum sequence length
            num_layers (int): Number of decoder layers
            num_heads (int): Number of heads in the multi-head attention
            d_ff (int): Hidden dimension of the Feed Forward Layer
            dropout_prob (float): Dropout probability
            pooling (Optional[str], optional): Pooling operation to be applied on the output of the encoder. Defaults
            to "mean"(average pooling), can also be "max"(max pooling), "cls"(use the first token of the sequence) or
            None(no pooling) - this is for seq to seq tasks.

        """
        super(GPTEncoder, self).__init__()

        self.embedding_dim = d_model
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.pooling = pooling

        self.embedding = TransformerEmbeddings(
            vocab_size=vocab_size_enc,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout_prob,
        )

        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_heads=num_heads,
                    dropout_prob=dropout_prob,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = LayerNorm(d_model)

        self.lm_head = nn.Linear(
            d_model, output_size
        )  # output_size is the required size of the output

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

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for the decoder of GPT model

        Args:
            src (torch.Tensor): Source sequence
            src_mask (torch.Tensor): Source mask

        Returns:
            torch.Tensor: Output of the decoder
        """

        # Embed the source sequence
        src = self.embedding(src)

        # Apply the decoder blocks
        for encoder_block in self.encoder_blocks:
            src = encoder_block(src, src_mask)

        # Apply the final layer norm
        src = self.final_norm(src)

        # Apply the pooling
        if self.pooling == "mean":
            src = torch.mean(src, dim=1)
        elif self.pooling == "max":
            src, _ = torch.max(src, dim=1)
        elif self.pooling == "cls":
            src = src[:, 0, :]
        elif self.pooling == "none":
            pass
        else:
            raise ValueError("Pooling should be either mean, max, cls or none")

        # Apply the linear layer
        src = self.lm_head(src)

        return src
