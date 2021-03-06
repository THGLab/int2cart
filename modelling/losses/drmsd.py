import torch
from modelling.utils.geometry import calc_dist_mat

def drmsd_loss(preds, batch, build_range):
    '''
    preds: predicted distance matrices [list of NxN matrices]
    batch: the batch dataset that contains target coordinates
    '''
    loss = 0
    n_total_entries = 0
    for batch_idx in range(len(batch.lengths)):
        length = batch.lengths[batch_idx]
        dist_mat = preds[batch_idx]

        target_coords = batch.crds[batch_idx].to(dist_mat.device)
        if build_range is None:
            target_res_idx = range(length)
        else:
            target_res_idx = range(*build_range[batch_idx])
            lenth = len(target_res_idx)
        backbone_indices = torch.cat([torch.arange(i*14,i*14+4) for i in target_res_idx])
        target_backbone_coords = target_coords[backbone_indices]
        target_dist_mat = calc_dist_mat(target_backbone_coords)

        batch_loss = torch.sum(torch.square(dist_mat - target_dist_mat))
        n_total_entries += ((4 * length) * (4 * length - 1))
        loss += batch_loss
    return loss / n_total_entries

