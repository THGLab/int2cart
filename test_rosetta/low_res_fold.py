
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
import pickle

sys.path.append("/home/jerry/data2/protein_building/nn_modelling/rosetta")
from int2cart_mover import Int2CartMover

post_stage_iter = 1000  # ~100 it/s for int2cart
frag_iter = 500000  # ~1800 it/s

total_iter = 100000

kT = 1.0
show_progress_bar = True
use_int2cart = False
nn_device = "cuda:0"

reference_pose = pose_from_pdb("1ubq.pdb")

def low_res_folding(pose, n_iter, mover, monte_carlo, save_file_name):
    scores_step = []
    scores_record = []
    rmsds = []
    gdttms = []
    it = range(n_iter)
    if show_progress_bar:
        it = tqdm(it, leave=False)
    for i in it:
        mover.apply(pose)
        scores_step.append(monte_carlo.score_function()(pose))
        monte_carlo.boltzmann(pose)
        scores_record.append(monte_carlo.score_function()(pose))
        rmsds.append(pyrosetta.rosetta.core.scoring.CA_rmsd(pose, reference_pose))
        gdttms.append(pyrosetta.rosetta.core.scoring.CA_gdtmm(pose, reference_pose))
    pose.dump_pdb(save_file_name)
    print("Current best score:", monte_carlo.score_function()(pose))
    return {"scores_step": scores_step,
    "scores_record": scores_record,
    "rmsds": rmsds,
    "gdttms": gdttms}
    

def main(seq, fragment_9mer, fragment_3mer, output_file_prefix, device=nn_device):
    """
    This program does low resolution folding
    """
    # initialize pose, set as centroid, and prepare scoring functions
    pose = pose_from_sequence(seq, res_type="centroid")
    # fa_scorefxn = get_score_function(True)
    ct_scorefxn = create_score_function("score3")

    int2cart_mover = Int2CartMover(seq, device=device)

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

    if use_int2cart:
        move_9_int2cart = SequenceMover()
        move_9_int2cart.add_mover(move_9mer)
        move_9_int2cart.add_mover(int2cart_mover)
        move_9mer = move_9_int2cart

        move_3_int2cart = SequenceMover()
        move_3_int2cart.add_mover(move_3mer)
        move_3_int2cart.add_mover(int2cart_mover)
        move_3mer = move_3_int2cart

        extra_mover = SequenceMover()
        extra_mover.add_mover(move_3mer)
        extra_mover.add_mover(int2cart_mover)

    else:
        extra_mover = move_3mer


    # low resolution monte carlo
    mc_low = MonteCarlo(pose, ct_scorefxn, kT)



    # start folding 
    # print("Low resolution folding...")
    # print("9-mer folding...")
    # _9mer_step_scores, _9mer_scores = low_res_folding(pose, frag_iter, move_9mer, mc_low, output_file_prefix + "_9mer.pdb")

    
    # print("3-mer folding...")
    # _3mer_step_scores, _3mer_scores = low_res_folding(pose, frag_iter, move_3mer, mc_low, output_file_prefix + "_3mer.pdb")


    # print("Final stage folding...")
    # start_time = datetime.now()
    # print("Start time:", start_time)
    # _final_step_scores, _final_scores = low_res_folding(pose, post_stage_iter, extra_mover, mc_low, output_file_prefix + "_final.pdb")
    # print("End time:", datetime.now())
    # print("Elapsed time:", datetime.now() - start_time)

    _9mer_results = low_res_folding(pose, total_iter, move_9mer, mc_low, output_file_prefix + "_9mer.pdb")
    _3mer_results = low_res_folding(pose, total_iter, move_3mer, mc_low, output_file_prefix + "_3mer.pdb")
        
    with open(output_file_prefix + "_scores.pkl", "wb") as f:
        pickle.dump({"9mer": _9mer_results,
        "3mer": _3mer_results}, f)

    # # switch to full atom and set up high resolution monte carlo
    # fa_switch = SwitchResidueTypeSetMover("fa_standard")
    # fa_switch.apply(pose)
    # mc_high = MonteCarlo(pose, fa_scorefxn, kT)

    # print("High resolution folding...")
    # for i in range(5):
    #     max_angle = 25 - 5 * i
    #     print("Max angle move:", max_angle)
    #     # progressively decrease max angle of small and shear movers
    #     small_mover.angle_max("H", max_angle)
    #     small_mover.angle_max("E", max_angle)
    #     small_mover.angle_max("S", max_angle)
    #     shear_mover.angle_max("H", max_angle)
    #     shear_mover.angle_max("E", max_angle)
    #     shear_mover.angle_max("S", max_angle)

    #     it = range(simulation_iter)
    #     if show_progress_bar:
    #         it = tqdm(it, leave=False)
    #     for _ in it:
    #         seq_mover.apply(pose)
    #         mc_high.boltzmann(pose)

    # print("Finished folding with score: ", fa_scorefxn(pose))
    # pose.dump_pdb(output_file_prefix + "_final.pdb")

if __name__ == "__main__": 
    folder = "ubiquitin_lowres"
    if not os.path.exists(folder):
        os.mkdir(folder)
    id = str(random.randrange(1e8))
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    if len(sys.argv) > 2:
        id = sys.argv[2]
    if len(sys.argv) > 3:
        device = sys.argv[3]
    else:
        device = nn_device
    if not os.path.exists(folder):
        os.makedirs(folder)
    # main("GSSGSSGTGVKPYGCSQCAKTFSLKSQLIVHQRSHTGVKPSGPSSG", "zinc_finger/aat000_09_05.200_v1_3", "zinc_finger/aat000_03_05.200_v1_3")
    main("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
     "ubiquitin/aat000_09_05.200_v1_3", "ubiquitin/aat000_03_05.200_v1_3", f"{folder}/{id}", device)
