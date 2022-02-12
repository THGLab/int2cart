import torch
from torch import nn
import numpy as np

class Standardize(nn.Module):
    r"""Standardize layer for shifting and scaling.

    .. math::
       y = \frac{x - \mu}{\sigma}

    Parameters
    ----------
    mean: torch.Tensor
        mean value :math:`\mu`.

    stddev: torch.Tensor
        standard deviation value :math:`\sigma`.

    eps: float, optional (default: 1e-9)
        small offset value to avoid zero division.

    Copyright: https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/nn/base.py
    """

    def __init__(self, mean, stddev, eps=1e-9):
        super(Standardize, self).__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
        self.register_buffer("eps", torch.ones_like(stddev) * eps)

    def forward(self, input):
        """Compute layer output.

        Parameters
        ----------
        input: torch.Tensor
            input data.

        Returns
        -------
        torch.Tensor: layer output.

        """
        # Add small number to catch divide by zero
        y = (input - self.mean) / (self.stddev + self.eps)
        return y


class ScaleShift(nn.Module):
    def __init__(self, mean, std, is_angle):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))
        self.register_buffer("is_angle", torch.tensor(is_angle))

    def forward(self, x):
        out = x * self.std + self.mean 
        if self.is_angle:
            out = torch.remainder((out + np.pi), 2 * np.pi) - np.pi
        return out
