from torch import nn
import torch

class VariableLengthBatchNorm(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(dimension)

    def forward(self, inputs, seq_lens):
        sequence_mask = torch.zeros(inputs.shape[:2]).bool()
        for i in range(len(seq_lens)):
            sequence_mask[i, :seq_lens[i]] = True
        batch_calculating_arr = inputs[sequence_mask]
        arr_after_bn = self.batch_norm(batch_calculating_arr)
        return_arr = torch.zeros_like(inputs)
        return_arr[sequence_mask] = arr_after_bn
        return return_arr