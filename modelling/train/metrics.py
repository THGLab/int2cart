import numpy as np
from modelling.utils import tensor_to_numpy
from Bio.SVDSuperimposer import SVDSuperimposer

def rmse(preds, targets, mask='auto', periodic=False):
    """Root mean square error"""
    assert preds.shape == targets.shape
    preds = tensor_to_numpy(preds)
    targets = tensor_to_numpy(targets)
    error = preds - targets
    if periodic:
        error = np.minimum(np.minimum(np.abs(error - 360), np.abs(error)), np.abs(error + 360))
    se = np.square(error)
    if mask == 'auto':
        mask = (targets != 0).astype(float)
    elif mask == 'none':
        mask = None
    if mask is not None:
        se *= mask
        return np.sqrt(np.sum(se) / (np.sum(mask) + 1e-7))
    else:
        return np.sqrt(np.mean(se))

def structure_rmsd(pred_coord_set, target_coord_set, seq_lens, calc_mean=True):
    rmsds = []
    pred_coord_set = tensor_to_numpy(pred_coord_set)
    target_coord_set = tensor_to_numpy(target_coord_set)
    for i in range(len(seq_lens)):
        pred_coord_i = pred_coord_set[i][:4 * seq_lens[i]]
        target_coord_i = target_coord_set[i][np.concatenate([np.arange(i*14,i*14+4) for i in range(seq_lens[i])])]
        superimposer = SVDSuperimposer()
        superimposer.set(pred_coord_i, target_coord_i)
        superimposer.run()
        rmsd = superimposer.get_rms()
        rmsds.append(rmsd)
    if calc_mean:
        return np.mean(rmsds)
    else:
        return rmsds