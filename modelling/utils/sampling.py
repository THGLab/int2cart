import torch
import numpy as np
from .torch_utils import tensor_to_numpy

def max_sampling(input_distribution, reference, data_format="numpy"):
    if data_format == "numpy":
        input_distribution = tensor_to_numpy(input_distribution)
        reference = tensor_to_numpy(reference)
        selected_indices = np.argmax(input_distribution, axis=-1)
        return reference[selected_indices]
    elif data_format == "tensor":
        selected_indices = torch.argmax(input_distribution, dim=-1)
        reference_torch = torch.tensor(reference).to(input_distribution.device)
        return reference_torch[selected_indices]