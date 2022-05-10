import torch
import numpy as np

AA_SEQ = "ACDEFGHIKLMNPQRSTVWY"
AA_MAP = {aa: idx for idx, aa in enumerate(AA_SEQ)}

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