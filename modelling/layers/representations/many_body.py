import torch
from torch import nn


def generate_grid(grid_length, grid_size, device):
    """
    The main function to create the outline of the grid cube for voxel representation.

    Parameters
    ----------
    grid_length: float or int
        The length of the edge of the grid box.

    grid_size: int
        number of vertices on each edge of the grid box.

    Returns
    -------
    torch.Tensor: The vertices of the grid box with shape: (G,G,G); G=grid_size

    """
    length = grid_length / 2.
    ticks = torch.linspace(-length, length, grid_size, device=device)
    x = ticks.repeat(grid_size,1).repeat(grid_size,1).t().contiguous().view(-1)
    y = ticks.repeat(grid_size,grid_size).t().contiguous().view(-1)
    z = ticks.repeat(grid_size ** 2)
    grid = torch.stack([x, y, z],dim=-1).view(grid_size,grid_size,grid_size,3)
    return grid


def gaussian(diff, sigma):
    r"""

    Parameters
    ----------
    diff: torch.Tensor
        This is the :math:`X - \mu` in the gaussian equation.

    sigma: torch.Tensor
        This is the standard deviation of gaussian distribution.

    Returns
    -------
    torch.Tensor: the gaussian distributions with shape: diff.shape[:-1]

    """
    diff = torch.norm(diff, 2, -1)
    coeff = -0.5 / torch.pow(sigma, 2)
    return torch.exp(coeff * torch.pow(diff, 2))


class ManyBodyVoxel(nn.Module):
    """
    The implementation of voxel representation as described by Liu et al.
    Todo: Reference

    Parameters
    ----------
    atom_types: list
        list of unique atomic numbers available to channelize representations.
        Each atomic number must be an integer.

    grid_length: list
        The length of the edge of the grid box.
        The grid_length must be passed as a list of one or more values.

    grid_size: int
        number of vertices on each edge of the grid box.

    sigma: torch.tensor
    the standard deviation of gaussian distribution.

    trainable_sigma: bool, optional (default: False)
        if True, sigma will be tuned during training.
    """

    def __init__(self, atom_types, grid_length, grid_size, sigma, trainable_sigma=False):
        super(ManyBodyVoxel, self).__init__()
        self.atom_types = atom_types
        self.grid_size = grid_size

        if not isinstance(grid_length, list):
            msg = 'The resolution must be passed as a list of one or more values.'
            raise TypeError(msg)
        self.grid_length = grid_length

        # sigma
        if trainable_sigma:
            self.register_parameter('sigma', sigma)
        else:
            self.register_buffer('sigma', sigma)


    def forward(self, distance_vector, atomic_numbers):
        """
        The main function to compute the voxel representation.

        Parameters
        ----------
        distance_vector: torch.Tensor
            A tensor of distance_vector with shape: (A/A', N, 3)

        atomic_numbers: torch.Tensor
            A tensor of atomic numbers for atoms in atomic environments.
            shape: (A/A', N)

        Returns
        -------
        torch.Tensor: voxel representation with shape: (A, C, G, G, G)
            where C refers to number of channels and is computed by (n_atom_types * n_grid_length)
            and G is exactly the value of grid_size parameter.
            Todo: we can support channels (C) at first or end, similar to the Conv layer.
        """

        representations = []
        for Z in self.atom_types:
            for L in self.grid_length:

                # generate grid
                grid = generate_grid(L, self.grid_size,
                                     device=distance_vector.device)                 # G, G, G, 3

                # atom type mask
                tmp_mask = torch.zeros_like(atomic_numbers)             # A, N
                tmp_mask[atomic_numbers == Z] = 1

                # atom-type-based distance vector
                tmp_dist_vect = distance_vector * tmp_mask[None,..., None]   # B, A, N, 3

                # voxel repr: 3D gaussian representation
                diff = grid[None, None, None, :, :, :, :] - tmp_dist_vect[:, :, :, None, None, None, :]
                box = gaussian(diff, self.sigma)                        # B, A, N, G, G, G
                assert box.shape == (distance_vector.shape[0], distance_vector.shape[1], distance_vector.shape[2],
                                     self.grid_size,self.grid_size,self.grid_size)

                # sum over neighbouring atoms
                box = torch.sum(box, dim=2)                             # B, A, G, G, G
                representations.append(box)
                del box

        representations = torch.stack(representations, dim=2)           # B, A, C, G, G, G

        return representations


class TwoBodyGaussian(nn.Module):
    r"""
    The implementation of gaussian/radial symmetry functions as described in the following papers:
        - Todo: Ref Parrinello
        - Todo: Ref Behler 2011, ACSF
        - Todo: Ref Schnet
        - Todo: Ref ANI

    The implementation is by the code for second reference: Todo: github repo

    Parameters
    ----------
        start: float, optional (default: 0.0)
            center of first Gaussian function, :math:`\mu_0`

        end: float, optional (default: 5.0)
            center of last Gaussian function, :math:`\mu_{N_g}`

        steps: int, optional (default: 50)
            total number of Gaussian functions, :math:`N_g`

        centered: bool, optional (default: False)
            If True, Gaussians are centered at the origin and
            the offsets are used as their widths (used e.g. for angular functions).

        trainable bool, optional (default: False)
            If True, widths and offset of Gaussian functions
            are adjusted during training process.

    """

    def __init__(
        self, start=0.0, end=5.0, steps=50, centered=False, trainable=False
    ):
        super(TwoBodyGaussian, self).__init__()

        # compute mean and standard deviation of Gaussian functions
        mu = torch.linspace(start=start, end=end, steps=steps)
        sigma = torch.FloatTensor((mu[1] - mu[0]) * torch.ones_like(mu))

        if trainable:
            self.register_parameter("mu", mu)
            self.register_parameter("sigma", sigma)
        else:
            self.register_buffer("mu", mu)
            self.register_buffer("sigma", sigma)

        self.centered = centered

    def forward(self, distances):
        """Compute smeared-gaussian distance values.

        Parameters
        ----------
            distances: torch.Tensor
                interatomic distance values of shape: (B, A, N)

        Returns
        -------
            torch.Tensor: layer output of shape: (B, A, N, G); G=steps

        """
        if self.centered:
            # if Gaussian functions are centered, use offsets to compute widths
            coeff = -0.5 / torch.pow(self.mu, 2)
            # if Gaussian functions are centered, no offset is subtracted
            diff = distances[:, :, :, None]
        else:
            # compute width of Gaussian functions (using an overlap of 1 STDDEV)
            coeff = -0.5 / torch.pow(self.sigma, 2)
            # Use advanced indexing to compute the individual components
            diff = distances[:, :, :, None] - self.mu[None, None, None, :]

        # compute smear distance values
        representation = torch.exp(coeff * torch.pow(diff, 2))

        return representation
