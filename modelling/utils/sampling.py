import numpy as np
from .torch_utils import tensor_to_numpy

def max_sampling(input_distribution, reference):
    input_distribution = tensor_to_numpy(input_distribution)
    reference = tensor_to_numpy(reference)
    selected_indices = np.argmax(input_distribution, axis=-1)
    return reference[selected_indices]