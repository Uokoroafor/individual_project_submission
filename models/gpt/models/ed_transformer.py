from typing import Optional, Dict
import torch
from torch import nn

from models.gpt.models.gpt_decoder import GPTDecoder
from models.gpt.models.gpt_encoder import GPTEncoder


class EncoderDecoderTransformer(nn.Module):
    def __init__(
            self,
            src_pad: int,
            src_sos: int,
            trg_pad: int,
            trg_sos: int,
            vocab_size_enc: int,
            vocab_size_dec: int,
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
        """Constructor class for the transformer. It consists of both the encoder and the decoder.
        Args:
            src_pad (int): Source padding index
            src_sos (int): Source start of sentence token
            trg_pad (int): Target padding index
            trg_sos (int): Target start of sentence token
            vocab_size_enc (int): Size of the vocabulary of the encoder
            vocab_size_dec (int): Size of the vocabulary of the decoder
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
        super(EncoderDecoderTransformer, self).__init__()
        self.src_pad = src_pad
        self.src_sos = src_sos
        self.trg_pad = trg_pad
        self.trg_sos = trg_sos
        self.vocab_size_enc = vocab_size_enc
        self.vocab_size_dec = vocab_size_dec
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

        self.decoder = GPTDecoder(
            vocab_size_dec=self.vocab_size_dec,
            d_model=self.d_model,
            max_seq_len=self.max_seq_len,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            dropout_prob=self.dropout_prob,
        )

        self.linear = nn.Linear(self.d_model, self.output_size)

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        """Forward pass of the transformer
        Args:
            src (torch.Tensor): Source tensor
            trg (torch.Tensor): Target tensor
        Returns:
            torch.Tensor: Output tensor
        """
        # src = [batch size, src len]
        # trg = [batch size, trg len]
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]
        enc_src = self.encoder(src, src_mask)
        # enc_src = [batch size, src len, d_model]
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        # output = [batch size, trg len, output dim]
        output = self.linear(output)
        # output = [batch size, trg len, output dim]
        return output

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create a mask for the source tensor
        Args:
            src (torch.Tensor): Source tensor
        Returns:
            torch.Tensor: Source mask tensor
        """
        # src = [batch size, src len]
        src_mask = (src != self.src_pad).unsqueeze(1).unsqueeze(2)
        # src_mask = [batch size, 1, 1, src len]
        return src_mask

    def make_trg_mask(self, trg: torch.Tensor) -> torch.Tensor:
        """Create a mask for the target tensor
        Args:
            trg (torch.Tensor): Target tensor
        Returns:
            torch.Tensor: Target mask tensor
        """
        # trg = [batch size, trg len]
        trg_pad_mask = (trg != self.trg_pad).unsqueeze(1).unsqueeze(2)
        # trg_pad_mask = [batch size, 1, 1, trg len]
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        # trg_sub_mask = [trg len, trg len]
        trg_mask = trg_pad_mask & trg_sub_mask
        # trg_mask = [batch size, 1, trg len, trg len]
        return trg_mask

    def greedy_decode(self, src: torch.Tensor, max_len: int = 50) -> torch.Tensor:
        """Greedy decoding
        Args:
            src (torch.Tensor): Source tensor
            max_len (int): Maximum length of the output
        Returns:
            torch.Tensor: Output tensor
        """
        # src = [batch size, src len]
        src_mask = self.make_src_mask(src)
        # src_mask = [batch size, 1, 1, src len]
        enc_src = self.encoder(src, src_mask)
        # enc_src = [batch size, src len, d_model]
        trg = torch.zeros((src.shape[0], 1), device=self.device).long()
        # trg = [batch size, 1]
        for i in range(max_len):
            trg_mask = self.make_trg_mask(trg)
            # trg_mask = [batch size, 1, i+1, i+1]
            output = self.decoder(trg, enc_src, trg_mask, src_mask)
            # output = [batch size, i+1, output dim]
            output = self.linear(output[:, -1])
            # output = [batch size, output dim]
            output = self.softmax(output)
            # output = [batch size, output dim]
            output = torch.argmax(output, dim=1).unsqueeze(1)
            # output = [batch size, 1]
            trg = torch.cat((trg, output), dim=1)
            # trg = [batch size, i+2]
        return trg[:, 1:]

    def beam_search_decode(self, src: torch.Tensor, max_len: int = 50, beam_size: int = 5) -> torch.Tensor:
        """Beam search decoding
        Args:
            src (torch.Tensor): Source tensor
            max_len (int): Maximum length of the output
            beam_size (int): Beam size
        Returns:
            torch.Tensor: Output tensor
        """
        # src = [batch size, src len]
        src_mask = self.make_src_mask(src)
        # src_mask = [batch size, 1, 1, src len]
        enc_src = self.encoder(src, src_mask)
        # enc_src = [batch size, src len, d_model]
        trg = torch.zeros((src.shape[0], 1), device=self.device).long()
        # trg = [batch size, 1]
        for i in range(max_len):
            trg_mask = self.make_trg_mask(trg)
            # trg_mask = [batch size, 1, i+1, i+1]
            output = self.decoder(trg, enc_src, trg_mask, src_mask)
            # output = [batch size, i+1, output dim]
            output = self.linear(output[:, -1])
            # output = [batch size, output dim]
            output = self.softmax(output)
            # output = [batch size, output dim]
            output = torch.topk(output, beam_size, dim=1)
            # output = [batch size, beam size]
            output = output[1]
            # output = [batch size, beam size]
            if i == 0:
                output = output.unsqueeze(1)
                # output = [batch size, 1, beam size]
            else:
                output = output.unsqueeze(1).repeat(1, i + 1, 1)
                # output = [batch size, i+1, beam size]
            trg = torch.cat((trg, output), dim=1)
            # trg = [batch size, i+2, beam size]
        return trg[:, 1:, 0]

    def translate(self, src: torch.Tensor, max_len: int = 50) -> torch.Tensor:
        """Translate the source tensor
        Args:
            src (torch.Tensor): Source tensor
            trg (torch.Tensor): Target tensor
            max_len (int): Maximum length of the output
        Returns:
            torch.Tensor: Output tensor
        """
        # src = [batch size, src len]
        # trg = [batch size, trg len]
        output = self.greedy_decode(src, max_len)
        # output = [batch size, trg len]
        return output

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
