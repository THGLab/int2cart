import numpy as np
from modelling.utils import tensor_to_numpy

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