import torch
from torch import nn


class ShellProvider(nn.Module):
    """
    This layer calculates distance of each atom in a molecule to its closest neighbouring atoms.

    Parameters
    ----------

    """
    def __init__(self):
        super(ShellProvider,self).__init__()

    def forward(self, atoms, neighbors,
                 cell=None, shift_vector=None, atom_idx=None, cutoff=None):
        """
        The main driver to calculate distances of atoms in a shell from center atom.

        Parameters
        ----------
        atoms: torch.Tensor
            XYZ coordinates of atoms in molecules.
            shape: (B, A, 3)

        neighbors: torch.Tensor
            index of neighbors of atoms.
            shape: (B, A/A', max_nbh)

        cell: torch.Tensor, optional (default: None)
            unit cell vectors. If cell is not None, shift_vector must be also provided.
            shape: (3, 3)

        shift_vector: torch.Tensor, optional (default: None)
            number of cell boundaries crossed by the bond between atom i and j
            (as presented by ASE library (ase.neighborlist module)
            shape: (B, A/A', N, 3)

        atom_idx: torch.Tensor, optional (default: None)
            Due to the memory explosion or any other considerations, only a few atoms in a molecule can
            be selected to be fed in to the next layers. The total number of atom_idx is the actual batch size.
            shape: (n_batch, n_atoms_masked)
            This is a list of indices, and if it's given, the batch size (n_batch) must be one.

        cutoff: float, optional (default: None)
            One last call to the cutoff for neighbour search. This helps to compile multiple resolutions, e.g., in a
            multi-channel model.

        Returns
        -------
        torch.Tensor: distance vector with shape: (B, A, N, 3)

        """
        # B: batch size
        # A: n_atoms
        # A': n_atoms_masked
        # N : max n_neighbours

        if atom_idx is not None:

            # sanity check
            if atom_idx.size()[0] != neighbors.size()[0] and neighbors.size()[0] > atoms.size()[0]:
                msg = '@ShellProvider: If atom_idx is provided, the first dimension of neighbors must be smaller than atoms.'
                raise ValueError(msg)

            # get positions of target atoms
            n_atoms = neighbors.shape[0]  # ==A'
            target_pos = atoms[atom_idx.view(n_atoms),:]              # A', 3
            # target_nbh = neighbours[atom_idx.view(n_atoms),:]             # A', N
            target_nbh_pos = atoms[neighbors, :]            # A', N, 3
            distance_vector = target_nbh_pos - target_pos[ :, None, :]    # A', N, 3

            # D = positions[j]-positions[i]+S.dot(cell)
            # S: shift_vector

            # add S.dot(cell)
            if cell is not None:        # 3, 3
                dim = shift_vector.size()
                shift = torch.bmm(shift_vector.view(dim[0], dim[1]*dim[2], dim[3]), cell)  # B, A'*N, 3
                shift = shift.view(dim)
                distance_vector = distance_vector + shift         # B, A', N, 3

            # distance = torch.norm(distance_vector, 2, 3)

        else:
            # get positions of target atoms
            n_atoms = neighbors.shape[0]  # ==A
            # target_nbh = neighbours[atom_idx.view(n_atoms),:]             # A, N
            target_nbh_pos = atoms[:, neighbors]  # B, A, N, 3
            distance_vector = target_nbh_pos - atoms[:, :, None, :]  # B, A, N, 3

            # D = positions[j]-positions[i]+S.dot(cell)
            # S: shift_vector

            # add S.dot(cell)
            if cell is not None:  # 3, 3
                dim = shift_vector.size()
                shift = torch.bmm(shift_vector.view(dim[0], dim[1] * dim[2], dim[3]), cell)  # B, A'*N, 3
                shift = shift.view(dim)
                distance_vector = distance_vector + shift  # B, A', N, 3

            # distance = torch.norm(distance_vector, 2, 3)

        return distance_vector


def atom_distances(
    positions,
    neighbors,
    cell=None,
    cell_offsets=None,
    return_vecs=False,
    normalize_vecs=False,
    neighbor_mask=None,
):
    r"""Compute distance of every atom to its neighbors.

    This function uses advanced torch indexing to compute differentiable distances
    of every central atom to its relevant neighbors.

    Args:
        positions (torch.Tensor):
            atomic Cartesian coordinates with (N_b x N_at x 3) shape
        neighbors (torch.Tensor):
            indices of neighboring atoms to consider with (N_b x N_at x N_nbh) shape
        cell (torch.tensor, optional):
            periodic cell of (N_b x 3 x 3) shape
        cell_offsets (torch.Tensor, optional) :
            offset of atom in cell coordinates with (N_b x N_at x N_nbh x 3) shape
        return_vecs (bool, optional): if True, also returns direction vectors.
        normalize_vecs (bool, optional): if True, normalize direction vectors.
        neighbor_mask (torch.Tensor, optional): boolean mask for neighbor positions.

    Returns:
        (torch.Tensor, torch.Tensor):
            distances:
                distance of every atom to its neighbors with
                (N_b x N_at x N_nbh) shape.

            dist_vec:
                direction cosines of every atom to its
                neighbors with (N_b x N_at x N_nbh x 3) shape (optional).

    """

    # Construct auxiliary index vector
    n_batch = positions.size()[0]
    idx_m = torch.arange(n_batch, device=positions.device, dtype=torch.long)[
        :, None, None
    ]
    # Get atomic positions of all neighboring indices
    pos_xyz = positions[idx_m, neighbors[:, :, :], :]

    # Subtract positions of central atoms to get distance vectors
    dist_vec = pos_xyz - positions[:, :, None, :]

    # add cell offset
    if cell is not None:
        B, A, N, D = cell_offsets.size()
        cell_offsets = cell_offsets.view(B, A * N, D)
        offsets = cell_offsets.bmm(cell)
        offsets = offsets.view(B, A, N, D)
        dist_vec += offsets

    # Compute vector lengths
    distances = torch.norm(dist_vec, 2, 3)

    if neighbor_mask is not None:
        # Avoid problems with zero distances in forces (instability of square
        # root derivative at 0) This way is neccessary, as gradients do not
        # work with inplace operations, such as e.g.
        # -> distances[mask==0] = 0.0
        tmp_distances = torch.zeros_like(distances)
        tmp_distances[neighbor_mask != 0] = distances[neighbor_mask != 0]
        distances = tmp_distances

    if return_vecs:
        tmp_distances = torch.ones_like(distances)
        tmp_distances[neighbor_mask != 0] = distances[neighbor_mask != 0]

        if normalize_vecs:
            dist_vec = dist_vec / tmp_distances[:, :, :, None]
        return distances, dist_vec
    return distances


class AtomDistances(nn.Module):
    r"""Layer for computing distance of every atom to its neighbors.

    Args:
        return_directions (bool, optional): if True, the `forward` method also returns
            normalized direction vectors.

    """

    def __init__(self, return_directions=False):
        super(AtomDistances, self).__init__()
        self.return_directions = return_directions

    def forward(
        self, positions, neighbors, cell=None, cell_offsets=None, neighbor_mask=None
    ):
        r"""Compute distance of every atom to its neighbors.

        Args:
            positions (torch.Tensor): atomic Cartesian coordinates with
                (N_b x N_at x 3) shape.
            neighbors (torch.Tensor): indices of neighboring atoms to consider
                with (N_b x N_at x N_nbh) shape.
            cell (torch.tensor, optional): periodic cell of (N_b x 3 x 3) shape.
            cell_offsets (torch.Tensor, optional): offset of atom in cell coordinates
                with (N_b x N_at x N_nbh x 3) shape.
            neighbor_mask (torch.Tensor, optional): boolean mask for neighbor
                positions. Required for the stable computation of forces in
                molecules with different sizes.

        Returns:
            torch.Tensor: layer output of (N_b x N_at x N_nbh) shape.

        """
        return atom_distances(
            positions,
            neighbors,
            cell,
            cell_offsets,
            return_vecs=self.return_directions,
            normalize_vecs=True,
            neighbor_mask=neighbor_mask,
        )


def triple_distances(
    positions,
    neighbors_j,
    neighbors_k,
    offset_idx_j=None,
    offset_idx_k=None,
    cell=None,
    cell_offsets=None,
):
    """
    Get all distances between atoms forming a triangle with the central atoms.
    Required e.g. for angular symmetry functions.

    Args:
        positions (torch.Tensor): Atomic positions
        neighbors_j (torch.Tensor): Indices of first neighbor in triangle
        neighbors_k (torch.Tensor): Indices of second neighbor in triangle
        offset_idx_j (torch.Tensor): Indices for offets of neighbors j (for PBC)
        offset_idx_k (torch.Tensor): Indices for offets of neighbors k (for PBC)
        cell (torch.tensor, optional): periodic cell of (N_b x 3 x 3) shape.
        cell_offsets (torch.Tensor, optional): offset of atom in cell coordinates
            with (N_b x N_at x N_nbh x 3) shape.

    Returns:
        torch.Tensor: Distance between central atom and neighbor j
        torch.Tensor: Distance between central atom and neighbor k
        torch.Tensor: Distance between neighbors

    """
    nbatch, _, _ = neighbors_k.size()
    idx_m = torch.arange(nbatch, device=positions.device, dtype=torch.long)[
        :, None, None
    ]

    pos_j = positions[idx_m, neighbors_j[:], :]
    pos_k = positions[idx_m, neighbors_k[:], :]

    if cell is not None:
        # Get the offsets into true cartesian values
        B, A, N, D = cell_offsets.size()

        cell_offsets = cell_offsets.view(B, A * N, D)
        offsets = cell_offsets.bmm(cell)
        offsets = offsets.view(B, A, N, D)

        # Get the offset values for j and k atoms
        B, A, T = offset_idx_j.size()

        # Collapse batch and atoms position for easier indexing
        offset_idx_j = offset_idx_j.view(B * A, T)
        offset_idx_k = offset_idx_k.view(B * A, T)
        offsets = offsets.view(B * A, -1, D)

        # Construct auxiliary aray for advanced indexing
        idx_offset_m = torch.arange(B * A, device=positions.device, dtype=torch.long)[
            :, None
        ]

        # Restore proper dmensions
        offset_j = offsets[idx_offset_m, offset_idx_j[:]].view(B, A, T, D)
        offset_k = offsets[idx_offset_m, offset_idx_k[:]].view(B, A, T, D)

        # Add offsets
        pos_j = pos_j + offset_j
        pos_k = pos_k + offset_k

    # if positions.is_cuda:
    #    idx_m = idx_m.pin_memory().cuda(async=True)

    # Get the real positions of j and k
    R_ij = pos_j - positions[:, :, None, :]
    R_ik = pos_k - positions[:, :, None, :]
    R_jk = pos_j - pos_k

    # + 1e-9 to avoid division by zero
    r_ij = torch.norm(R_ij, 2, 3) + 1e-9
    r_ik = torch.norm(R_ik, 2, 3) + 1e-9
    r_jk = torch.norm(R_jk, 2, 3) + 1e-9

    return r_ij, r_ik, r_jk


class TriplesDistances(nn.Module):
    """
    Layer that gets all distances between atoms forming a triangle with the
    central atoms. Required e.g. for angular symmetry functions.
    """

    def __init__(self):
        super(TriplesDistances, self).__init__()

    def forward(self, positions, neighbors_j, neighbors_k):
        """
        Args:
            positions (torch.Tensor): Atomic positions
            neighbors_j (torch.Tensor): Indices of first neighbor in triangle
            neighbors_k (torch.Tensor): Indices of second neighbor in triangle

        Returns:
            torch.Tensor: Distance between central atom and neighbor j
            torch.Tensor: Distance between central atom and neighbor k
            torch.Tensor: Distance between neighbors

        """
        return triple_distances(positions, neighbors_j, neighbors_k)


def neighbor_elements(atomic_numbers, neighbors):
    """
    Return the atomic numbers associated with the neighboring atoms. Can also
    be used to gather other properties by neighbors if different atom-wise
    Tensor is passed instead of atomic_numbers.

    Args:
        atomic_numbers (torch.Tensor): Atomic numbers (Nbatch x Nat x 1)
        neighbors (torch.Tensor): Neighbor indices (Nbatch x Nat x Nneigh)

    Returns:
        torch.Tensor: Atomic numbers of neighbors (Nbatch x Nat x Nneigh)

    """
    # Get molecules in batch
    n_batch = atomic_numbers.size()[0]
    # Construct auxiliary index
    idx_m = torch.arange(n_batch, device=atomic_numbers.device, dtype=torch.long)[
        :, None, None
    ]
    # Get neighbors via advanced indexing
    neighbor_numbers = atomic_numbers[idx_m, neighbors[:, :, :]]
    return neighbor_numbers


class NeighborElements(nn.Module):
    """
    Layer to obtain the atomic numbers associated with the neighboring atoms.
    """

    def __init__(self):
        super(NeighborElements, self).__init__()

    def forward(self, atomic_numbers, neighbors):
        """
        Args:
            atomic_numbers (torch.Tensor): Atomic numbers (Nbatch x Nat x 1)
            neighbors (torch.Tensor): Neighbor indices (Nbatch x Nat x Nneigh)

        Returns:
            torch.Tensor: Atomic numbers of neighbors (Nbatch x Nat x Nneigh)
        """
        return neighbor_elements(atomic_numbers, neighbors)
