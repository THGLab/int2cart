import torch
from torch import nn
from modelling.utils.default_scalers import *
import numpy as np

cross_entropy = nn.CrossEntropyLoss(reduction='none')

def categorical_loss(raw_predictions, raw_targets, digitizer):
    device = raw_predictions.device
    mask = (raw_targets != 0).float().to(device)
    target_labels = digitizer(raw_targets).to(device)
    loss = cross_entropy(raw_predictions, target_labels)
    masked_loss = loss * mask
    return masked_loss

def numerical_loss(raw_predictions, raw_targets, scaler):
    device = raw_predictions.device
    raw_targets = raw_targets.to(device)
    mask = (raw_targets != 0).float().to(device)
    if type(scaler) is not float:
        scaler = torch.tensor(scaler).to(device).reshape((1, 1,) + (len(scaler),))
    rescaled_difference = (raw_predictions - raw_targets) / scaler
    masked_loss = mask * rescaled_difference ** 2
    return masked_loss

def prepare_losses(settings, angle_digitizer, n_ca_blens_digitizer, ca_c_blens_digitizer, c_n_blens_digitizer, rescale_by_length=False, central_residue=None):
    
    loss_term_weights = settings['training']['loss_term_weigths']
    if type(loss_term_weights) is not list:
        loss_term_weights = [loss_term_weights] * 14



    def compute_loss(preds, batch, sample_weights=None):
        '''
        preds: list of predictions [3x backbone bond angles, 6x sidechain torsion angles, 3x bond lengths]
        batch: the batch dataset that contains targets and masks
        '''
        device = preds[0].device
        
        # angle losses

        backbone_angle_preds = torch.cat(preds[:3], axis=-1)
        backbone_angle_targets = batch.angs[:, :, 3:6]
        masked_backbone_angle_loss = numerical_loss(backbone_angle_preds, backbone_angle_targets, 
        [default_stds["N_CA_C_bond_angle"], default_stds["CA_C_N_bond_angle"], default_stds["C_N_CA_bond_angle"]])
        
        sidechain_torsion_preds = torch.moveaxis(torch.stack(preds[3:9], axis=-1), 2, 1)
        sidechain_torsion_targets = batch.angs[:, :, 6:] * 180 / np.pi
        masked_sidechain_torsion_loss = categorical_loss(sidechain_torsion_preds, sidechain_torsion_targets, angle_digitizer)

        masked_angle_loss = torch.cat([masked_backbone_angle_loss, masked_sidechain_torsion_loss], dim=-1)

       
        # bond length losses
        blens_preds = torch.cat(preds[9:12], axis=-1)
        blens_targets = batch.blens
        masked_blens_loss = numerical_loss(blens_preds, blens_targets, 
        [default_stds["N_CA_bond_length"], default_stds["CA_C_bond_length"], default_stds["C_N_bond_length"]])

        # sidechain bond length losses
        sc_blens_preds = preds[12]
        sc_blens_targets = batch.sc_blens
        masked_sc_blens_loss = numerical_loss(sc_blens_preds, sc_blens_targets, default_stds["CA_CB_bond_length"])

        # sidechain bond angle losses
        sc_ang_preds = preds[13]
        sc_ang_targets = batch.sc_angs
        masked_sc_ang_loss = numerical_loss(sc_ang_preds, sc_ang_targets, default_stds["N_CA_CB_bond_angle"])
        
        # masked total loss
        mask = batch.msks.to(device)
        weighted_loss = torch.sum(torch.cat([masked_angle_loss, masked_blens_loss, \
            masked_sc_blens_loss, masked_sc_ang_loss], dim=-1) * \
            torch.tensor(loss_term_weights).to(device), dim=-1)
        if rescale_by_length:
            length_scaler = torch.tensor(batch.lengths, device=device)[:, None] / 100
            weighted_loss = weighted_loss * length_scaler
        if central_residue is not None:
            weighted_loss = weighted_loss[:, central_residue]
            mask = mask[:, central_residue]
        if sample_weights is not None:
            mask = mask * sample_weights
        loss = torch.sum((weighted_loss * mask) / torch.sum(mask + 1e-8))

        # print('batch total loss:', loss.detach().cpu().numpy())

        return loss

    return compute_loss