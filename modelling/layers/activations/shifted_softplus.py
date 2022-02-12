import numpy as np
from torch import nn


def shifted_softplus(x):
    r"""Compute shifted soft-plus activation function.
    As it is used in the https://github.com/atomistic-machine-learning/schnetpack

    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Args:
        x (torch.Tensor): input tensor.

    Returns:
        torch.Tensor: shifted soft-plus of input.

    """
    return nn.functional.softplus(x) - np.log(2.0)