from torch import nn, Tensor
from typing import Iterable
import torch


class FCNetwork(nn.Module):
    """Fully connected PyTorch neural network class with batch normalization

    :attr input_size (int): dimensionality of input tensors
    :attr out_size (int): dimensionality of output tensors
    :attr layers (torch.nn.Module): neural network as sequential network of multiple layers
    """

    def __init__(self, dims: Iterable[int], output_activation: nn.Module = None, hidden_activation: nn.Module = nn.ReLU, use_batch_norm: bool = False):
        """Creates a network using specified activation between layers and optional output activation

        :param dims (Iterable[int]): tuple in the form of (IN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2,
            ..., OUT_SIZE) for dimensionalities of layers
        :param output_activation (nn.Module): PyTorch activation function to use after last layer
        :param hidden_activation (nn.Module): PyTorch activation function to use between layers
        :param use_batch_norm (bool): whether to use batch normalization
        """
        super().__init__()
        self.input_size = dims[0]
        self.out_size = dims[-1]
        self.use_batch_norm = use_batch_norm
        self.layers = self.make_seq(dims, output_activation, hidden_activation, use_batch_norm)

    @staticmethod
    def make_seq(dims: Iterable[int], output_activation: nn.Module, hidden_activation: nn.Module, use_batch_norm: bool = False) -> nn.Module:
        """Creates a sequential network using specified activation between layers and optional output activation

        :param dims (Iterable[int]): tuple in the form of (IN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2,
            ..., OUT_SIZE) for dimensionalities of layers
        :param output_activation (nn.Module): PyTorch activation function to use after last layer
        :param hidden_activation (nn.Module): PyTorch activation function to use between layers
        :param use_batch_norm (bool): whether to use batch normalization
        :return (nn.Module): return created sequential layers
        """
        mods = []

        for i in range(len(dims) - 2):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            if use_batch_norm:
                mods.append(nn.BatchNorm1d(dims[i + 1], track_running_stats=False))
            mods.append(hidden_activation())

        mods.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation:
            mods.append(output_activation())
        return nn.Sequential(*mods)

    def forward(self, x: Tensor) -> Tensor:
        """Computes a forward pass through the network

        :param x (torch.Tensor): input tensor to feed into the network
        :return (torch.Tensor): output computed by the network
        """
        if self.use_batch_norm and x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        return self.layers(x)

    def hard_update(self, source: nn.Module):
        """Updates the network parameters by copying the parameters of another network

        :param source (nn.Module): network to copy the parameters from
        """
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, source: nn.Module, tau: float):
        """Updates the network parameters with a soft update

        Moves the parameters towards the parameters of another network

        :param source (nn.Module): network to move the parameters towards
        :param tau (float): stepsize for the soft update
            (tau = 0: no update; tau = 1: copy parameters of source network)
        """
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(
                (1 - tau) * target_param.data + tau * source_param.data
            )
