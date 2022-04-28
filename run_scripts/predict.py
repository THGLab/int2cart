'''
Use a trained model to predict backbone bond lengths and bond angles, and optionally build the structure from the predictions.

Author: Jie Li
Date created: Apr 27, 2022
'''

import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from modelling.models.builder import BackboneBuilder

AA_SEQ = "ACDEFGHIKLMNPQRSTVWY"
AA_MAP = {aa: idx for idx, aa in enumerate(AA_SEQ)}
ONE_TO_THREE_LETTER_MAP = {
    "R": "ARG",
    "H": "HIS",
    "K": "LYS",
    "D": "ASP",
    "E": "GLU",
    "S": "SER",
    "T": "THR",
    "N": "ASN",
    "Q": "GLN",
    "C": "CYS",
    "G": "GLY",
    "P": "PRO",
    "A": "ALA",
    "V": "VAL",
    "I": "ILE",
    "L": "LEU",
    "M": "MET",
    "F": "PHE",
    "Y": "TYR",
    "W": "TRP"
}

def predict(builder, seq, tors, build=True, units="degree", device="cpu"):
    '''
    builder: A BackboneBuilder object for making predictions and build protein structures
    seq: the one-letter amino acid sequence of the protein to be predicted
    tors: An Nx3 numpy array specifying the backbone torsion angles in units of rad
        N is sequence length (should be the same as the length of seq)
        3 corresponds to phi, psi, omega
    units: the units for the angles, either degree or radian. This applies to both inputs and outputs
    device: the device used to run the model, can be a GPU device (CUDA:x) or CPU 

    returns:
    A dictionary containing d1, d2, d3, theta1, theta2, theta3 for the predictions of the current protein
    optionally the coordinates (when build=True)
    '''
    # do unit conversion when necessary: model accepts radians
    if units == "degree":
        tors = torch.tensor(tors).to(device).float() * np.pi/180
    elif units == "radian":
        tors = torch.tensor(tors).to(device).float()
    else:
        raise RuntimeError("Torsion angle units should be one of [degree] or [radian]")

    # prepare inputs and make predictions
    res_type = torch.tensor([[AA_MAP[aa] for aa in seq]]).to(device)
    model_inputs = {"phi": tors[None, :, 0],
                    "psi": tors[None, :, 1],
                    "omega": tors[None, :, 2],
                    "res_type": res_type,
                    "lengths": [len(seq)]}
    if build:
        results = builder(model_inputs)
        predictions = results["predictions"]
        coords = results["coords"].detach().cpu().numpy()
    else:
        predictions = builder.predictor(model_inputs)

    # convert data types back to numpy arrays, and do unit conversion when necessary: model predicts radians
    for key in predictions:
        if key.startswith("d"):
            predictions[key] = predictions[key].detach().cpu().numpy()
        elif key.startswith("theta"):
            predictions[key] = predictions[key].detach().cpu().numpy() * \
                (180/np.pi if units == "degree" else 1)
    if build:
        return predictions, coords
    else:
        return predictions

def main(model_addr, seq, tors_addr, output_name, build=True, units="degree", device="cpu", model_config="../configs/predict.yml"):
    settings = yaml.safe_load(open(model_config, "r"))
    builder = BackboneBuilder(settings)
    model_state = torch.load(model_addr)["model_state_dict"]
    builder.load_predictor_weights(model_state)
    builder.to(device)
    tors = np.load(tors_addr)
    if np.isnan(tors).any():
        print("Warning! NaN found in torsion input array. Treated as zero.")
        tors = np.nan_to_num(tors)
    if build:
        predictions, coords = predict(builder, seq, tors, build, units, device)
        generate_pdb(seq, coords, output_name + ".pdb")
    else:
        predictions = predict(builder, seq, tors, build, units, device)
    print_geometry(seq, tors, predictions, output_name + "_geom.csv")

def generate_pdb(seq, coords, output_file):
    pdb_lines = []
    atom_idx = 0
    for resid, aa in enumerate(seq):
        res_type = ONE_TO_THREE_LETTER_MAP[aa]
        for atom in ['N', 'CA', 'C', 'O']:
            pdb_lines.append('ATOM   {:4d}  {:<2s}  {} A{:4d}     {:>7.3f} {:>7.3f} {:>7.3f}  1.00  0.00           {}  '.format(
                atom_idx + 1, atom, res_type, resid + 1, 
                coords[0, atom_idx, 0], coords[0, atom_idx, 1], coords[0, atom_idx, 2], atom[0] 
            ))
            atom_idx += 1
    
    with open(output_file, "w") as f:
        f.write("\n".join(pdb_lines))

def print_geometry(seq, tors, preds, output_file):
    with open(output_file, "w") as f:
        f.write("SeqID, AA, phi, psi, omega, d1(N-CA), d2(CA-C), d3(C-N), theta1(N-CA-C), theta2(CA-C-N), theta3(C-N-CA)\n")
        for i in range(len(seq)):
            f.write("{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n".format(
                i + 1, ONE_TO_THREE_LETTER_MAP[seq[i]], tors[i, 0], tors[i, 1], tors[i, 2],
                preds['d1'][0][i][0], preds['d2'][0][i][0], preds['d3'][0][i][0],
                preds['theta1'][0][i][0], preds['theta2'][0][i][0], preds['theta3'][0][i][0],
            ))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Int2Cart, predicting backbone bond lengths and bond angles " + 
        "from a protein amino acid sequence and the backbone phi, psi, omega torsion angle profiles.\n" + 
        "It can also build backbone structures using predicted bond lengths and bond angles.")
    parser.add_argument("seq", help="the one-letter amino acid sequence of the protein to be predicted")
    parser.add_argument("torsion_addr", help="the address to a numpy file that stores " + 
        "the backbone torsion angles for the protein to be predicted. Expected to be in size of " + 
        "Nx3, where N is the sequence length, and 3 corresponds to phi, psi and omega.")
    parser.add_argument("output", help="The output file name, excluding file extension")
    parser.add_argument("--build", "-b", action="store_true", help="build the structure after predicting bond lengths and bond angles")
    parser.add_argument("--units", "-u", default="degree", help="the units for angles used for inputs and outputs." + 
        " Should be one of [degree](default) or [radian]")
    parser.add_argument("--device", "-d", default="cpu", help="the device used for running the model, could be " + 
        "[cpu](default) or [cuda:x] for GPU")
    parser.add_argument("--config", "-c", default="../configs/predict.yml", help="configuration file specifying model structure")
    parser.add_argument("--model_addr", "-m", default="../trained_models/release.tar", help="a path specifying the position of trained model")
    args = parser.parse_args()
    main(args.model_addr, args.seq, args.torsion_addr, args.output, args.build, args.units, args.device, args.config)


