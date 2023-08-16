from typing import Optional, Dict
import torch
from torch import nn
import torch.nn.functional as F
from models.gpt.models.gpt_decoder import GPTDecoder
import torch.distributions as dist


class GPT(nn.Module):
    def __init__(
        self,
        trg_pad: int,
        trg_sos: int,
        vocab_size_dec: int,
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
            trg_pad (int): Target padding index
            trg_sos (int): Target start of sentence token
            vocab_size_dec (int): Size of the vocabulary of the decoder
            d_model (int): Dimension of the model
            d_ff (int): Hidden dimension of the Feed Forward Layer
            max_seq_len (int): Maximum sequence length
            num_layers (int): Number of decoder layers
            num_heads (int): Number of heads in the multi-head attention
            dropout_prob (float): Dropout probability
            device (str): Device - 'cpu' or 'cuda'
        """
        super(GPT, self).__init__()
        self.trg_pad = trg_pad
        self.trg_sos = trg_sos
        self.decoder = GPTDecoder(
            vocab_size_dec,
            d_model,
            max_seq_len,
            num_layers,
            num_heads,
            d_ff,
            dropout_prob,
        )
        self.device = device
        self.max_seq_len = max_seq_len

    def forward(self, trg: torch.Tensor) -> torch.Tensor:
        """Forward pass of the transformer
        Args:
            trg (torch.Tensor): Target tensor of shape (batch_size, seq_len)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, vocab_size)
        """
        trg_mask = self.get_trg_mask(trg)
        decoder_output = self.decoder(trg, trg_mask)
        return decoder_output

    def get_trg_mask(self, trg: torch.Tensor) -> torch.Tensor:
        """Create target mask
        Args:
            trg (torch.Tensor): Target tensor of shape (batch_size, seq_len)
        Returns:
            torch.Tensor: Target mask tensor of shape (batch_size, seq_len, seq_len)
        """
        # Want to ignore the padding tokens
        trg_pad_mask = (trg != self.trg_pad).unsqueeze(-2)  # (batch_size, 1, seq_len)
        trg_len = trg.shape[1]
        # What to ignore the future tokens (i.e. tokens that are not yet predicted)
        trg_sub_mask = torch.tril(
            torch.ones((trg_len, trg_len), device=self.device)
        ).bool()
        # Final mask ignores both padding and future tokens
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def generate(
        self,
        start_token: int,
        max_length: int,
        sampled: Optional[bool] = True,
        k: Optional[int] = 5,
        temp: Optional[float] = 1.0,
    ) -> torch.Tensor:
        """
        Generate a sequence given a start token
        Args:
            start_token (int): Start token
            max_length (int): Maximum length of the sequence to generate
            sampled (bool): Whether to sample from the output distribution or take the argmax (greedy)
            k (int): Number of top-k tokens to sample from
            temp (float): Temperature to apply to the output logits

        Returns:
            torch.Tensor: Generated sequence
        """
        assert max_length > 0, "Maximum length must be greater than 0"
        generated = torch.tensor(
            [start_token], dtype=torch.long, device=self.device
        ).unsqueeze(0)
        out = torch.tensor(
            [start_token], dtype=torch.long, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            for j in range(max_length):
                if generated.shape[1] == max_length:
                    break
                if generated.shape[1] > self.max_seq_len:
                    # If the generated sequence is longer than the maximum length, truncate it
                    generated = generated[:, -self.max_seq_len :]
                output = self.forward(generated)

                if sampled:
                    # apply a temperature to the output logits
                    assert temp > 0.0, "Temperature must be greater than 0.0"
                    assert (
                        k > 0
                    ), "k, the number of top-k tokens, must be greater than 0"
                    # output = output[:, -1, :] / temp
                    output /= temp

                    # Apply top-k filtering
                    v, _ = torch.topk(output, k)
                    output[output < v[:, :, [-1]]] = -float("Inf")

                    # apply a softmax to transform the logits to probabilities
                    probabilities = F.softmax(output[:, -1, :], dim=-1)
                    categorical_dist = dist.Categorical(probs=probabilities)
                    next_token = categorical_dist.sample()

                    # Need to make it a 2D tensor
                    next_token = next_token.unsqueeze(1)

                    assert round(probabilities.sum().item(), 2) == 1.0, (
                        f"Probabilities do not sum to 1.0,"
                        f" instead sum to {probabilities.sum().item()}"
                    )
                else:
                    next_token = output.argmax(2)[:, -1].unsqueeze(1)

                generated = torch.cat((generated, next_token), dim=1)
                out = torch.cat((out, next_token), dim=1)
            return out

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
