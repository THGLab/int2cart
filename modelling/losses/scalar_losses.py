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

def prepare_losses(settings, angle_digitizer, n_ca_blens_digitizer, ca_c_blens_digitizer, c_n_blens_digitizer, rescale_by_length=False):
    
    loss_term_weights = settings['training']['loss_term_weigths']
    if type(loss_term_weights) is not list:
        loss_term_weights = [loss_term_weights] * 12



    def compute_loss(preds, batch):
        '''
        preds: list of predictions [3x backbone bond angles, 6x sidechain torsion angles, 3x bond lengths]
        batch: the batch dataset that contains targets and masks
        '''
        device = preds[0].device
        
        # angle losses
        if settings['bins']['backbone_angle_bin']:
            angle_pred_raw = torch.moveaxis(torch.stack(preds[:9], axis=-1), 2, 1)
            angle_targets = batch.angs[:, :, 3:] * 180 / np.pi
            masked_angle_loss = categorical_loss(angle_pred_raw, angle_targets, angle_digitizer)
        else:
            backbone_angle_preds = torch.cat(preds[:3], axis=-1)
            backbone_angle_targets = batch.angs[:, :, 3:6]
            masked_backbone_angle_loss = numerical_loss(backbone_angle_preds, backbone_angle_targets, 
            [default_stds["N_CA_C_bond_angle"], default_stds["CA_C_N_bond_angle"], default_stds["C_N_CA_bond_angle"]])
            
            sidechain_torsion_preds = torch.moveaxis(torch.stack(preds[3:9], axis=-1), 2, 1)
            sidechain_torsion_targets = batch.angs[:, :, 6:] * 180 / np.pi
            masked_sidechain_torsion_loss = categorical_loss(sidechain_torsion_preds, sidechain_torsion_targets, angle_digitizer)

            masked_angle_loss = torch.cat([masked_backbone_angle_loss, masked_sidechain_torsion_loss], dim=-1)

       
        # bond length losses
        if settings['bins']['bond_length_bin']:
            n_ca_blens_pred_raw = torch.moveaxis(preds[9], -1, 1)
            n_ca_blens_targets = batch.blens[:, :, 0]
            n_ca_blens_loss = categorical_loss(n_ca_blens_pred_raw, n_ca_blens_targets, n_ca_blens_digitizer)

            ca_c_blens_pred_raw = torch.moveaxis(preds[10], -1, 1)
            ca_c_blens_targets = batch.blens[:, :, 1]
            ca_c_blens_loss = categorical_loss(ca_c_blens_pred_raw, ca_c_blens_targets, ca_c_blens_digitizer)

            c_n_blens_pred_raw = torch.moveaxis(preds[11], -1, 1)
            c_n_blens_targets = batch.blens[:, :, 2]
            c_n_blens_loss = categorical_loss(c_n_blens_pred_raw, c_n_blens_targets, c_n_blens_digitizer)
            
            masked_blens_loss = torch.stack([n_ca_blens_loss, ca_c_blens_loss, c_n_blens_loss], dim=-1)
        else:
            blens_preds = torch.cat(preds[9:], axis=-1)
            blens_targets = batch.blens
            masked_blens_loss = numerical_loss(blens_preds, blens_targets, 
            [default_stds["N_CA_bond_length"], default_stds["CA_C_bond_length"], default_stds["C_N_bond_length"]])
        
        # masked total loss
        mask = batch.msks.to(device)
        weighted_loss = torch.sum(torch.cat([masked_angle_loss, masked_blens_loss], dim=-1) * \
            torch.tensor(loss_term_weights).to(device), dim=-1)
        if rescale_by_length:
            length_scaler = torch.tensor(batch.lengths, device=device)[:, None] / 100
            weighted_loss = weighted_loss * length_scaler
        loss = torch.sum((weighted_loss * mask) / torch.sum(mask + 1e-8))

        # print('batch total loss:', loss.detach().cpu().numpy())

        return loss

    return compute_loss