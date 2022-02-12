import torch
from torch import nn


def gaussian_smearing(distances, offset, widths, centered=False):
    r"""Smear interatomic distance values using Gaussian functions.

    Args:
        distances (torch.Tensor): interatomic distances of (N_b x N_at x N_nbh) shape.
        offset (torch.Tensor): offsets values of Gaussian functions.
        widths: width values of Gaussian functions.
        centered (bool, optional): If True, Gaussians are centered at the origin and
            the offsets are used to as their widths (used e.g. for angular functions).

    Returns:
        torch.Tensor: smeared distances (N_b x N_at x N_nbh x N_g).

    """
    if not centered:
        # compute width of Gaussian functions (using an overlap of 1 STDDEV)
        coeff = -0.5 / widths ** 2
        # Use advanced indexing to compute the individual components
        diff = distances[:, :, None] - offset[None, None, :]
    else:
        # if Gaussian functions are centered, use offsets to compute widths
        coeff = -0.5 / offset ** 2
        # if Gaussian functions are centered, no offset is subtracted
        diff = distances[:, :, None]
    # compute smear distance values
    gauss = torch.exp(coeff * diff ** 2)
    return gauss


class GaussianSmearing(nn.Module):
    r"""Smear layer using a set of Gaussian functions.

    Parameters
    ----------
    start: (float, optional)
        center of first Gaussian function, :math:`\mu_0`.
    stop: (float, optional)
        center of last Gaussian function, :math:`\mu_{N_g}`
    n_gaussians: (int, optional)
        total number of Gaussian functions, :math:`N_g`.

    width_factor: float, optional (default: 1.5)
        adjust the SD of gaussians.
        this is a constant factor multiplied by the bin width

    centered: (bool, optional)
        If True, Gaussians are centered at the origin and
        the offsets are used to as their widths (used e.g. for angular functions).
    margin: int
        The margin helps with the symmetric labels like angles. The margin specifies the
        number of bins to transfer to the head/tail of the bins from the other end.
        if zero, it will be skipped.

    normalize: bool, optional (default: False)
        if normalize final output of gaussians (divide by sum)

    """

    def __init__(
            self, start=0.0, stop=5.0, n_gaussians=50, margin=0,
            width_factor=1.5, centered=False, normalize=False
    ):
        super(GaussianSmearing, self).__init__()
        # add margin
        self.margin = margin
        if margin > 0 :
            extra_domain = (stop - start)/n_gaussians * margin
            # self.upper_limit = stop - extra_domain
            # self.lower_limit = start + extra_domain
            start -= extra_domain
            stop += extra_domain
            n_gaussians += int(2*margin)


        # compute offset and width of Gaussian functions
        offsets = torch.linspace(start, stop, n_gaussians)
        widths = width_factor * (offsets[1] - offsets[0]) * torch.ones_like(offsets)

        self.register_buffer('offsets', offsets)
        self.register_buffer('widths', widths)
        self.centered = centered
        self.normalize = normalize

    def forward(self, features):
        """Compute smeared-gaussian distance values.

        Parameters
        ----------
        features: ndarray
            raw feature values of (batch_size, n_features) shape.

        Returns
        -------
        torch.Tensor: layer output of (batch_size, n_features, n_gaussians) shape.

        """
        x = gaussian_smearing(
            features, self.offsets, self.widths, centered=self.centered
        )

        if self.margin > 0:

            # mask_right = features>= self.upper_limit
            x[:,:, self.margin:2*self.margin] += x[:,:, -self.margin:]


            # mask_left = features<= self.lower_limit
            x[:,:, -2*self.margin:-self.margin] += x[:,:, :self.margin]

            x = x[:,:, self.margin:-self.margin]

        if self.normalize:
            x = x / torch.sum(x, axis=-1)[..., None]

        return x
