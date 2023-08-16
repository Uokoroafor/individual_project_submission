from typing import Tuple, Optional
import torch
from torch import nn
from models.gpt.layers.attention import Attention


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        """Constructor class for the SelfAttention layer this is the same as the CausalSelfAttention layer except that it does not have the trg_mask argument

        Args:
            d_model (int): Dimension of the model
            num_heads (int): Number of heads
        """
        super(SelfAttention, self).__init__()

        # Check if the d_model is divisible by the number of heads
        assert (
            d_model % num_heads == 0
        ), "d_model must be divisible by the number of heads"

        # Set the d_model and num_heads
        self.d_model = d_model
        self.num_heads = num_heads

        # Set the depth of each head
        self.depth = d_model // num_heads

        # Create the query, key, value and linear layers
        self.query_layer = nn.Linear(d_model, d_model)
        self.key_layer = nn.Linear(d_model, d_model)
        self.value_layer = nn.Linear(d_model, d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

        # Create the attention layer
        self.attention = Attention()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the CausalSelfAttention layer
        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_model)
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_model)
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_model)
            src_mask (Optional[torch.Tensor], optional): Mask tensor of shape (batch_size, seq_len, seq_len). Defaults to None.
        Returns:
            torch.Tensor: SelfAttention output of shape (batch_size, seq_len, d_model)
            torch.Tensor: Attention weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        # Get the batch size
        batch_size = query.shape[0]

        # Pass the query, key and value through their respective linear layers
        query = self.query_layer(query)
        key = self.key_layer(key)
        value = self.value_layer(value)

        # Reshape the query, key and value
        # The reshaping is necessary to divide the `d_model` dimensions into `num_heads` number of heads,
        # each having `depth` dimensions.
        query = query.reshape(batch_size, -1, self.num_heads, self.depth)
        key = key.reshape(batch_size, -1, self.num_heads, self.depth)
        value = value.reshape(batch_size, -1, self.num_heads, self.depth)

        # Transpose the query, key and value
        # The transpose is necessary to perform the matrix multiplication in the attention calculation
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        seq_len = query.size(-2)

        # Create the mask
        if src_mask is None:
            mask = torch.ones(seq_len, seq_len)
        else:
            mask = src_mask
            # replicate the mask over the number of heads
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        mask = mask.float()
        mask = mask.to(query.device)

        # Calculate the attention using the query, key and value
        attention, attention_weights = self.attention(query, key, value, mask)

        # Transpose the attention output
        # The transpose is necessary to concatenate the multi-heads
        attention = attention.transpose(1, 2)

        # Concatenate the multi-heads
        # The reshape is necessary to combine the `num_heads` and `depth` dimensions
        concat_attention = attention.reshape(batch_size, -1, self.d_model)

        # Pass the concatenated attention through the linear layer
        output = self.linear_layer(concat_attention)

        return output, attention_weights
