import numpy as np

def tensor_to_numpy(tensor):
    if type(tensor) is np.ndarray:
        return tensor
    return  tensor.detach().cpu().numpy()