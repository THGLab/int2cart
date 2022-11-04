'''
A custom movemap that probablistically selects residues to be movable based on b-factor or pLDDT score.
Author: Jie Li
Date created: Nov 2, 2022
'''

import pyrosetta
import numpy as np

def get_bfactor_vector(pose: pyrosetta.Pose):
    """
    Return a selection vector based on b-factors.
    above = get all above. So to select bad b-factors above is ``True``,
    but to select AF2 bad ones. above is ``False``
    """
    pdb_info = pose.pdb_info()
    vector = []
    for r in range(1, pose.total_residue() + 1):
        try:
            atom_index = pose.residue(r).atom_index('CA')
        except AttributeError:
            atom_index = 1
        bfactor = pdb_info.bfactor(r, atom_index)
        vector.append(bfactor)
    return vector

def converter(x, low, high):
    '''piecewise conversion function
        when x is smaller than low, return 1
        when x is larger than high, return 0
        when x is between low and high, return a linear function
    '''
    if x < low:
        return 1
    elif x > high:
        return 0
    else:
        return (high - x) / (high - low)

class BFactorMoveMap():
    def __init__(self, pose, low=50, high=90, random_seed=42):
        self.b_factors = get_bfactor_vector(pose)
        self.low = low
        self.high = high
        self.random = np.random.RandomState(random_seed)
        self.move_chances = np.array([converter(x, low, high) for x in self.b_factors])
        self.move_chances = self.move_chances / self.move_chances.sum()
        self.accumulated_chances = np.cumsum(self.move_chances)
        self.move_map = pyrosetta.MoveMap()

    def sample(self):
        """
        Sample an index from accumulated chances
        """
        r = self.random.rand()
        return np.searchsorted(self.accumulated_chances, r)

    def advance(self, print_index=False):
        """
        Advance the move map
        """
        self.move_map.clear()
        index = self.sample()
        self.move_map.set_bb(index + 1, True)
        self.move_map.set_bb(index + 2, True)
        if print_index:
            print(index + 1, index + 2)