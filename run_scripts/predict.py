'''
Use a trained model to predict backbone bond lengths and bond angles, and optionally build the structure from the predictions.

Author: Jie Li
Date created: Apr 27, 2022
'''

import argparse
import yaml
import torch
import numpy as np
from modelling.models.builder import BackboneBuilder
from modelling.utils.predict import predict

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

def main(model_addr, seq, tors, output_name, build=True, output_geom=True, units="degree", device="cpu", model_config="../configs/predict.yml"):
    settings = yaml.safe_load(open(model_config, "r"))
    builder = BackboneBuilder(settings)
    model_state = torch.load(model_addr)["model_state_dict"]
    builder.load_predictor_weights(model_state)
    builder.to(device)
    if np.isnan(tors).any():
        print("Warning! NaN found in torsion input array. Treated as zero.")
        tors = np.nan_to_num(tors)
    if build:
        predictions, coords = predict(builder, seq, tors, build, units, device)
        generate_pdb(seq, coords, output_name + ".pdb")
    else:
        predictions = predict(builder, seq, tors, build, units, device)
    if output_geom:
        print_geometry(seq, tors, predictions, output_name + "_geom.csv")

def generate_pdb(seq, coords, output_file):
    pdb_lines = []
    atom_idx = 0
    for resid, aa in enumerate(seq):
        res_type = ONE_TO_THREE_LETTER_MAP[aa]
        for atom in ['N', 'CA', 'C', 'O']:
            pdb_lines.append('ATOM   {:4d}  {:<2s}  {} A{:4d}    {:>8.3f}{:>8.3f}{:>8.3f}  1.00  0.00           {}  '.format(
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
    parser.add_argument("seq", nargs='?', default=None, help="the one-letter amino acid sequence of the protein to be predicted")
    parser.add_argument("torsion_addr", nargs='?', default=None, help="the address to a numpy file that stores " + 
        "the backbone torsion angles for the protein to be predicted. Expected to be in size of " + 
        "Nx3, where N is the sequence length, and 3 corresponds to phi, psi and omega.")
    parser.add_argument("-output", nargs='?', default=None, help="The output file name, excluding file extension")
    parser.add_argument("--from_pdb", default=None, help="extract amino acid sequence and torsion angles from the pdb file " + 
     "instead of providing them through arguments")
    parser.add_argument("--build", "-b", action="store_true", help="build the structure after predicting bond lengths and bond angles")
    parser.add_argument("--units", "-u", default="degree", help="the units for angles used for inputs and outputs." + 
        " Should be one of [degree](default) or [radian]")
    parser.add_argument("--device", "-d", default="cpu", help="the device used for running the model, could be " + 
        "[cpu](default) or [cuda:x] for GPU")
    parser.add_argument("--config", "-c", default="../configs/predict.yml", help="configuration file specifying model structure")
    parser.add_argument("--model_addr", "-m", default="../trained_models/release.tar", help="a path specifying the position of trained model")
    args = parser.parse_args()

    output_path = args.output
    # extract sequence and torsion angles if provided a pdb file
    if args.from_pdb is not None:
        with open(args.from_pdb) as f:
            pdb_lines = f.readlines()
        from pdbtools.torsion import pdbTorsion
        from pdbtools.seq import pdbSeq2Fasta
        seq = pdbSeq2Fasta(pdb_lines, full_fasta=False)
        tors = np.nan_to_num(pdbTorsion(pdb_lines)[0])
        if output_path is None:
            output_path = args.from_pdb.replace(".pdb", "_processed")
    else:
        seq = args.seq
        tors = np.load(args.tors_addr)
    main(args.model_addr, seq, tors, output_path, args.build, True, args.units, args.device, args.config)


