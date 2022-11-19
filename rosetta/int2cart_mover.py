"""
Author: Jie Li
Date created: Oct 18, 2022

This script defines a pyrosetta mover that modifies bond lengths and bond angles of a pose using int2cart predictions
"""

import os
import yaml
import torch
import numpy as np
from modelling.models import get_model
import pyrosetta
from pyrosetta import AtomID

AA_SEQ = "ACDEFGHIKLMNPQRSTVWY"
AA_MAP = {aa: idx for idx, aa in enumerate(AA_SEQ)}

file_path = os.path.dirname(os.path.realpath(__file__))
default_model_config = os.path.join(file_path, "../configs/predict.yml")
default_model_path = os.path.join(file_path, "../trained_models/release.tar")

class Int2CartMover(pyrosetta.rosetta.protocols.moves.Mover):
    def __init__(self, seq, n_move=None, device="cpu", model_addr=default_model_path, model_config=default_model_config) -> None:
        pyrosetta.rosetta.protocols.moves.Mover.__init__(self)
        # Create model and load state dict
        settings = yaml.safe_load(open(model_config, "r"))
        self.predictor = get_model(settings)
        self.predictor.load_state_dict(torch.load(model_addr, map_location=torch.device('cpu'))["model_state_dict"])
        self.predictor.to(device)
        self.predictor.eval()
        self.device = device
        self.n_move = n_move  # how many residues of bond lengths and bond angles to modify

        # Save residue types as torch tensor to be reused
        self.seq_len = len(seq)
        self.res_type = torch.tensor([[AA_MAP[aa] for aa in seq]]).to(device)

    def get_torsion_from_pose(self, pose):
        phis = [pose.phi(i) for i in range(1, self.seq_len + 1)]
        psis = [pose.psi(i) for i in range(1, self.seq_len + 1)]
        omegas = [pose.omega(i) for i in range(1, self.seq_len + 1)]
        return phis, psis, omegas

    def predict(self, phis, psis, omegas):
        # Make appropriate conversions for phi, psi and omega inputs
        phis = torch.tensor(phis).to(self.device).float() * np.pi/180
        psis = torch.tensor(psis).to(self.device).float() * np.pi/180
        omegas = torch.tensor(omegas).to(self.device).float() * np.pi/180

        # Prepare inputs and make predictions
        model_inputs = {"phi": phis[None, :],
                        "psi": psis[None, :],
                        "omega": omegas[None, :],
                        "res_type": self.res_type,
                        "lengths": [self.seq_len]}
        predictions = self.predictor(model_inputs)

        # Convert data types back to numpy arrays, note angles are in radians
        for key in predictions:
            predictions[key] = predictions[key].detach().cpu().numpy()[0]
        return predictions


    def apply(self, pose, residues_to_move=None):
        phis, psis, omegas = self.get_torsion_from_pose(pose)
        predictions = self.predict(phis, psis, omegas)
        # prepare the residues to move based on n_move
        if residues_to_move is None:
            residues_to_move = range(1, self.seq_len + 1)
            if self.n_move is not None:
                residues_to_move = np.random.choice(residues_to_move, self.n_move, replace=False)
        # Modify bond lengths and bond angles using predictions
        conf = pose.conformation()
        for i in range(1, self.seq_len + 1):
            if i not in residues_to_move:
                continue
            # Define the AtomIDs to be used
            N = AtomID(pose.residue(i).atom_index("N"), i)
            CA = AtomID(pose.residue(i).atom_index("CA"), i)
            C = AtomID(pose.residue(i).atom_index("C"), i)
            if i != self.seq_len:
                Np1 = AtomID(pose.residue(i + 1).atom_index("N"), i + 1)
                CAp1 = AtomID(pose.residue(i + 1).atom_index("CA"), i + 1)

            # Modify bond lengths
            conf.set_bond_length(N, CA, predictions["d1"][i - 1])
            conf.set_bond_length(CA, C, predictions["d2"][i - 1])
            if i != self.seq_len:
                conf.set_bond_length(C, Np1, predictions["d3"][i - 1])

            # Modify bond angles
            conf.set_bond_angle(N, CA, C, predictions["theta1"][i - 1])
            if i != self.seq_len:
                conf.set_bond_angle(CA, C, Np1, predictions["theta2"][i - 1])
                conf.set_bond_angle(C, Np1, CAp1, predictions["theta3"][i - 1])

        
if __name__ == "__main__":
    # Test code
    from pyrosetta import *
    init()
    
    pose = pose_from_pdb("../../pdbs/1ubq.pdb")
    mover = Int2CartMover(pose.sequence(), device="cpu", n_move=10)
    pose.dump_pdb("../../pdbs/before_move.pdb")
    mover.apply(pose, residues_to_move=[15,50])
    pose.dump_pdb("../../pdbs/after_move_15_50.pdb")
