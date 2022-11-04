import pyrosetta; pyrosetta.init()
from pyrosetta import *
from pyrosetta.teaching import *
import numpy as np
import sys
sys.path.append("/home/jerry/data2/protein_building/nn_modelling/rosetta")

from scorefxn import RelRgScore, VDWScore, CompositeScore, rg_scorefxn
from monte_carlo import CustomMonteCarlo
from movemaps import BFactorMoveMap
import pandas as pd
from tqdm import tqdm

structure = "/home/jerry/data2/protein_building/human_proteins/AF-Q99680-F1-model_v3.pdb"
if len(sys.argv) > 1:
    structure = sys.argv[1]

pose = pyrosetta.pose_from_pdb(structure)

kT = 1.0
n_moves = 1
max_move_angle_H = 0
max_move_angle_E = 2.5
max_move_angle_L = 3

n_total_iters = 10000 # 100000
output_freq = 1000
show_progress_bar = True

output_prefix = "Q99680"

scorefxn = CompositeScore([RelRgScore(), VDWScore()], [100, 1])

bfactor_movemap = BFactorMoveMap(pose)

small_mover = SmallMover(bfactor_movemap.move_map, kT, n_moves)
shear_mover = ShearMover(bfactor_movemap.move_map, kT, n_moves)


small_mover.angle_max("H", max_move_angle_H)
small_mover.angle_max("E", max_move_angle_E)
small_mover.angle_max("L", max_move_angle_L)
shear_mover.angle_max("H", max_move_angle_H)
shear_mover.angle_max("E", max_move_angle_E)
shear_mover.angle_max("L", max_move_angle_L)

seq_mover = SequenceMover()
seq_mover.add_mover(small_mover) 
seq_mover.add_mover(shear_mover)

mc = CustomMonteCarlo(pose, scorefxn, kT)

total_scores = []
vdw_scores = []
rg_scores = []
rg_values = []

iterator = range(n_total_iters)
if show_progress_bar:
    iterator = tqdm(iterator, leave=False)
for n in iterator:
    bfactor_movemap.advance()
    small_mover.apply(pose)
    bfactor_movemap.advance()
    shear_mover.apply(pose)
    mc.boltzmann(pose)
    if n % output_freq == 0:
        total_score = scorefxn(pose)
        vdw_score = scorefxn.scorefxns[1](pose)
        rg_score = scorefxn.scorefxns[0](pose)
        rg_value = rg_scorefxn(pose)

        total_scores.append(total_score)
        vdw_scores.append(vdw_score)
        rg_scores.append(rg_score)
        rg_values.append(rg_value)

pose.dump_pdb(output_prefix + "_final.pdb")
df = pd.DataFrame({"total_score": total_scores, "vdw_score": vdw_scores, "rg_score": rg_scores, "rg_value": rg_values})
df.to_csv(output_prefix + "_scores.csv")
