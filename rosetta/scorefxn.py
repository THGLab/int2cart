'''
Some custom score functions for Rosetta.

Author: Jie Li
Date created: Nov 2, 2022
'''

import pyrosetta
from pyrosetta import *
import numpy as np
import pickle
from scipy import stats
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

# use the Rg scoring function term in rosetta to calculate protein Rg
rg_scorefxn = ScoreFunction()
rg_scorefxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.rg, 1)

# Relateiv Rg statistics data from PISCES dataset
with open(os.path.join(dir_path, "rg_data.pkl"), "rb") as f:
    rg_data=pickle.load(f)

# use kernel density estimation to fit the relative Rg distribution
kernel = stats.gaussian_kde(rg_data,bw_method=0.2)

def calc_rel_rg_score(pose):
    '''
    The expected Rg has the form Rg=3 * N^(1/3) according to 
    Kolinski, A.; Godzik, A.; Skolnick, J. A general method for 
    the prediction of the three dimensional structure and folding 
    pathway of globular proteins: Application to designed helical 
    proteins. The Journal of Chemical Physics 1993, 98 (9), 7420-7433.
    and Dima, R. I.; Thirumalai, D. Asymmetry in the Shapes of 
    Folded and Denatured States of Proteins. The Journal of Physical 
    Chemistry B 2004, 108 (21), 6564-6570.
    '''
    rg = rg_scorefxn(pose)
    seq_len = len(pose.sequence())
    ratio = rg / (3 * seq_len**(1/3))
    return -np.log(kernel(ratio)[0] + 1e-8)

class RelRgScore(pyrosetta.rosetta.core.scoring.ScoreFunction):
    '''
    An empirical negative log likelihood score function based on relative Rg
    '''
    def __init__(self, weight=1) -> None:
        pyrosetta.rosetta.core.scoring.ScoreFunction.__init__(self)
        self.weight = weight

    def __call__(self, pose):
        return self.weight * calc_rel_rg_score(pose)

class VDWScore(pyrosetta.rosetta.core.scoring.ScoreFunction):
    '''
    van der Waals energy is the sum of the attractive and repulsive components.
    '''
    def __init__(self, weight=(1,1)) -> None:
        pyrosetta.rosetta.core.scoring.ScoreFunction.__init__(self)
        self.scorefxn = ScoreFunction()
        self.scorefxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.fa_atr, weight[0])
        self.scorefxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.fa_rep, weight[1])


    def __call__(self, pose):
        return self.scorefxn(pose)

class CompositeScore(pyrosetta.rosetta.core.scoring.ScoreFunction):
    '''
    A composite score function
    '''
    def __init__(self, scorefxns, weights) -> None:
        pyrosetta.rosetta.core.scoring.ScoreFunction.__init__(self)
        self.scorefxns = scorefxns
        self.weights = weights

    def __call__(self, pose):
        score = 0
        for i in range(len(self.scorefxns)):
            score += self.weights[i] * self.scorefxns[i](pose)
        return score