
"""
This program does ab initio folding
"""

import pyrosetta
pyrosetta.init()
from pyrosetta import *
from pyrosetta.teaching import *
from pyrosetta.rosetta.core.fragment import *
from tqdm import tqdm
from datetime import datetime
import os
import random
import sys

simulation_iter = 1000
frag_iter = 500000
kT = 1.0
show_progress_bar = False

def low_res_folding(pose, move_9, move_3, monte_carlo, save_file_prefix):
    print("9-mer folding...")
    it = range(frag_iter)
    if show_progress_bar:
        it = tqdm(it, leave=False)
    for _ in it:
        move_9.apply(pose)
        monte_carlo.boltzmann(pose)
    pose.dump_pdb(save_file_prefix + "_9mer.pdb")
    print("9-mer best score:", monte_carlo.score_function()(pose))
    
    print("3-mer folding...")
    for _ in it:
        move_3.apply(pose)
        monte_carlo.boltzmann(pose)
    pose.dump_pdb(save_file_prefix + "_3mer.pdb")
    print("3-mer best score:", monte_carlo.score_function()(pose))
        
    monte_carlo.show_state()

def main(seq, fragment_9mer, fragment_3mer, output_file_prefix):
    """
    This program does ab inito folding
    """
    # initialize pose, set as centroid, and prepare scoring functions
    pose = pose_from_sequence(seq, res_type="centroid")
    fa_scorefxn = get_score_function(True)
    ct_scorefxn = create_score_function("score3")

    # initiate fragment set
    fragmentSet9 = ConstantLengthFragSet(9)
    fragmentSet3 = ConstantLengthFragSet(3)
    fragmentSet9.read_fragment_file(fragment_9mer)
    fragmentSet3.read_fragment_file(fragment_3mer)

    # set up movemap and fragment mover
    movemap = MoveMap()
    movemap.set_bb(True)
    move_9mer = ClassicFragmentMover(fragmentSet9, movemap)
    move_3mer = ClassicFragmentMover(fragmentSet3, movemap)

    # low resolution monte carlo
    mc_low = MonteCarlo(pose, ct_scorefxn, kT)

    #set up small and shear movers    
    n_moves = 5
    small_mover = SmallMover(movemap, kT, n_moves)
    shear_mover = ShearMover(movemap, kT, n_moves)
    
    #set up minimize mover
    min_mover = MinMover()
    min_mover.movemap(movemap)
    min_mover.score_function(fa_scorefxn)
    min_mover.min_type("linmin")
    min_mover.tolerance(0.5)

    
    #set up sequence mover and repeat mover
    seq_mover = SequenceMover()
    seq_mover.add_mover(small_mover) 
    seq_mover.add_mover(min_mover)
    seq_mover.add_mover(shear_mover)
    seq_mover.add_mover(min_mover)

    # low resolution folding 
    print("Low resolution folding...")
    low_res_folding(pose, move_9mer, move_3mer, mc_low, output_file_prefix)

    # switch to full atom and set up high resolution monte carlo
    fa_switch = SwitchResidueTypeSetMover("fa_standard")
    fa_switch.apply(pose)
    mc_high = MonteCarlo(pose, fa_scorefxn, kT)

    print("High resolution folding...")
    for i in range(5):
        max_angle = 25 - 5 * i
        print("Max angle move:", max_angle)
        # progressively decrease max angle of small and shear movers
        small_mover.angle_max("H", max_angle)
        small_mover.angle_max("E", max_angle)
        small_mover.angle_max("S", max_angle)
        shear_mover.angle_max("H", max_angle)
        shear_mover.angle_max("E", max_angle)
        shear_mover.angle_max("S", max_angle)

        it = range(simulation_iter)
        if show_progress_bar:
            it = tqdm(it, leave=False)
        for _ in it:
            seq_mover.apply(pose)
            mc_high.boltzmann(pose)

    print("Finished folding with score: ", fa_scorefxn(pose))
    pose.dump_pdb(output_file_prefix + "_final.pdb")

if __name__ == "__main__": 
    folder = "ubiquitin_ab_initio"
    if not os.path.exists(folder):
        os.mkdir(folder)
    id = str(random.randrange(1e8))
    if len(sys.argv) > 1:
        output_name = sys.argv[1]
    if len(sys.argv) > 2:
        id = sys.argv[2]
    # main("GSSGSSGTGVKPYGCSQCAKTFSLKSQLIVHQRSHTGVKPSGPSSG", "zinc_finger/aat000_09_05.200_v1_3", "zinc_finger/aat000_03_05.200_v1_3")
    main("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
     "ubiquitin/aat000_09_05.200_v1_3", "ubiquitin/aat000_03_05.200_v1_3", f"{folder}/{id}")
