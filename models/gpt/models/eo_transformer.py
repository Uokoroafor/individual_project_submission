from typing import Optional, Dict
import torch
from torch import nn
from models.gpt.models.gpt_encoder import GPTEncoder


class EncodeOnlyTransformer(nn.Module):
    def __init__(
        self,
        src_pad: int,
        src_sos: int,
        vocab_size_enc: int,
        output_size: int,
        pooling: str,
        d_model: int,
        d_ff: int,
        max_seq_len: int,
        num_layers: Optional[int] = 6,
        num_heads: Optional[int] = 8,
        dropout_prob: Optional[float] = 0.1,
        device: Optional[str] = "cpu",
    ):
        """Constructor class for the Encoder Only transformer. It consists of just the encoder.
        Args:
            src_pad (int): Source padding index
            src_sos (int): Source start of sentence token
            vocab_size_enc (int): Size of the vocabulary of the encoder
            output_size (int): Size of the output
            pooling (str): Pooling type - 'cls' or 'mean'
            d_model (int): Dimension of the model
            d_ff (int): Hidden dimension of the Feed Forward Layer
            max_seq_len (int): Maximum sequence length
            num_layers (int): Number of decoder layers
            num_heads (int): Number of heads in the multi-head attention
            dropout_prob (float): Dropout probability
            device (str): Device - 'cpu' or 'cuda'
        """
        super(EncodeOnlyTransformer, self).__init__()
        self.src_pad = src_pad
        self.src_sos = src_sos
        self.vocab_size_enc = vocab_size_enc
        self.output_size = output_size
        self.pooling = pooling
        self.d_model = d_model
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.device = device

        self.encoder = GPTEncoder(
            vocab_size_enc=self.vocab_size_enc,
            output_size=self.output_size,
            d_model=self.d_model,
            max_seq_len=self.max_seq_len,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            dropout_prob=self.dropout_prob,
            pooling=self.pooling,
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Forward pass of the transformer
        Args:
            src (torch.Tensor): Source tensor of shape (batch_size, seq_len)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, output_size)
        """
        src_mask = self.get_src_mask(src)
        encoder_output = self.encoder(src, src_mask)
        return encoder_output

    def get_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create source mask
        Args:
            src (torch.Tensor): Source tensor of shape (batch_size, seq_len)
        Returns:
            torch.Tensor: Source mask tensor of shape (batch_size, seq_len, seq_len)
        """
        src_pad_mask = (src != self.src_pad).unsqueeze(-2)

        # Expand mask tensor to shape (batch_size, seq_len, seq_len)
        src_mask = src_pad_mask.expand(-1, src.size(1), -1)

        return src_mask

    def count_parameters(self) -> Dict[str, int]:
        """Counts the parameters of the model and returns a dictionary
        Returns:
            Dict[str, int]: Dictionary of the parameter counts
        """
        counts = {}
        for name, module in self.named_modules():
            if isinstance(module, nn.Module):
                count = sum(p.numel() for p in module.parameters() if p.requires_grad)

                # format the counts to have commas for thousands
                count = "{:,}".format(count)
                counts[name] = count
        # Change name of the first key to total and format the value to integer
        counts["total"] = int(counts.pop("").replace(",", ""))
        return counts
