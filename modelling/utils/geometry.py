import torch

def calc_dist_mat(input_coords):
    '''
    Helper function for formulating pairwise distance matrix from point coordinates in 3d.
    Based on D[i,j] = (a[i] - a[j])(a[i] - a[j])' = r[i] - 2 * a[i] * a[j]' + r[j], where r[i] is the squared norm of the ith point
    Ref: https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow 

    args:
        input_coords = coordinates of the points (torch.Tensor <(N, 3)>)

    returns:
        pairwise distance matrix (torch.Tensor <(N, N)>)
    '''
    r = torch.sum(input_coords * input_coords, dim=-1, keepdim=True)
    D = r - 2 * torch.matmul(input_coords, input_coords.T) + r.T
    return torch.sqrt(torch.clamp(D, 1e-8))