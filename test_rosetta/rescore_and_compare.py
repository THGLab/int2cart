from pyrosetta import *
from pyrosetta.rosetta import *
pyrosetta.init()

from glob import glob
from tqdm import tqdm
import os
import pandas as pd

switch = SwitchResidueTypeSetMover("centroid")
scorefxn = create_score_function("score3")
native_pose=pose_from_pdb("1ubq.pdb")

def calc_metrics(pdb_filename):
    pose = pose_from_pdb(pdb_filename)
    switch.apply(pose)
    score = scorefxn(pose)
    rmsd = pyrosetta.rosetta.core.scoring.CA_rmsd(pose, native_pose)
    gdtmm = pyrosetta.rosetta.core.scoring.CA_gdtmm(pose, native_pose)
    return score, rmsd, gdtmm

_pdbs=glob("./lowres_int2cart/*.pdb")

names=[]
scores=[]
rmsds=[]
gdttms=[]
for pdb in tqdm(_pdbs):
    score, rmsd, gdttm = calc_metrics(pdb)
    scores.append(score)
    rmsds.append(rmsd)
    gdttms.append(gdttm)
    names.append(os.path.basename(pdb))

df = pd.DataFrame({"name":names, "score":scores, "rmsd":rmsds, "gdttm":gdttms})
df.to_csv("lowres_int2cart.csv", index=False)
print("finish!")