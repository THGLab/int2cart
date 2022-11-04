'''
A customly written Monte Carlo class for Rosetta

Author: Jie Li
Date created: Nov 2, 2022
'''

import numpy as np

class CustomMonteCarlo:
    def __init__(self, pose, scorefxn, kT, random_seed=None) -> None:
        self.last_accepted_pose = pose.clone()
        self.last_accepted_score = scorefxn(pose)
        self.kT = kT
        self.scorefxn = scorefxn
        self.random = np.random.RandomState(random_seed)

    def boltzmann(self, pose):
        score = self.scorefxn(pose)
        criterion = np.exp(-(score - self.last_accepted_score)/self.kT)
        succeed = self.random.rand() < criterion
        if succeed:
            self.last_accepted_pose = pose.clone()
            self.last_accepted_score = score
        else:
            pose.assign(self.last_accepted_pose)
        return succeed