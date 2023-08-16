from typing import List, Optional

import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int = 1,
                 activation: Optional[nn.Module] = None):
        """This class is used to create a neural network with the specified number of hidden layers and hidden units.

        Args:

            input_size (int): The number of input features
            hidden_sizes (List[int]): A list containing the number of hidden units in each hidden layer
            output_size (int): The number of output features - defaults to 1
            activation (Optional[nn.Module], optional): The activation function to be used. Defaults to None.
        """

        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        if activation is None:
            self.activation = nn.ReLU()
        self.activation = activation
        self.layers = nn.ModuleList()
        self._build_layers()

    def _build_layers(self):
        """This method is used to build the neural network layers"""
        sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method is used to compute the forward pass of the neural network

        Args:
            x (torch.Tensor): The input tensor

        Returns:
            torch.Tensor: The output tensor
        """
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x

    def count_parameters(self) -> int:
        """This method is used to count the number of trainable parameters in the neural network

        Returns:
            int: The number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
