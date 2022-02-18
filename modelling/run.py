import yaml
import sys
import torch
from torch import nn
from torch.optim import Adam
import numpy as np

from torch.nn import Sequential
from modelling.layers import ScaleShift
from modelling.models import RecurrentModel, MLP, MultiHeadModel
from modelling.utils import OnehotDigitizer
from sidechainnet import load
from modelling.train import Trainer
from modelling.utils.default_scalers import *
from modelling.utils.get_gpu import handle_gpu

#
settings_path = 'configs/debug.yml'
debug_mode = True
if len(sys.argv) > 1:
    settings_path = sys.argv[-1]
    debug_mode = False
settings = yaml.safe_load(open(settings_path, "r"))

device = [torch.device(dev) for dev in handle_gpu(settings['general']['device'])]


# data
data = load(settings['data']['casp_version'],
            thinning=settings['data']['thinning'],
            with_pytorch='dataloaders',
            scn_dir=settings['data']['scn_data_dir'],
            batch_size=settings['training']['batch_size'])

train = data['train']
val = data[f'valid-{settings["data"]["validation_similarity_level"]}']
test = data['test']

# model
smearing = {'start': -180,
            'stop': 180,
            'n_gaussians': settings['model']['n_gaussians'],
            'margin': settings['model']['gaussian_margin'],
            'width_factor': settings['model']['gaussian_factor'],
            'normalize': settings['model']['gaussian_normalize']}

latent_dim = settings['model']['rec_neurons_num'] 
hidden_dim = latent_dim // 2

rnn_encoder = RecurrentModel(recurrent=settings['model']['recurrent'],
                            smearing_parameters=smearing,
                            n_filter_layers=settings['model']['n_filter_layers'],
                            filter_size=settings['model']['filter_size'],
                            res_embedding_size=settings['model']['filter_size'],
                            rec_stack_size=settings['model']['rec_stack_size'],
                            rec_neurons_num=settings['model']['rec_neurons_num'] // 2,  # bidirectional
                            rec_dropout=settings['model']['rec_dropout'])

# prepare output heads
if settings['bins']['bond_length_bin']:
    n_ca_blens_predictor = MLP(latent_dim, settings['bins']['n-ca_bin'][-1], n_hidden=hidden_dim)
    ca_c_blens_predictor = MLP(latent_dim, settings['bins']['ca-c_bin'][-1], n_hidden=hidden_dim)
    c_n_blens_predictor = MLP(latent_dim, settings['bins']['c-n_bin'][-1], n_hidden=hidden_dim)
else:
    n_ca_blens_predictor = Sequential(MLP(latent_dim, 1, n_hidden=hidden_dim), 
        ScaleShift(default_means["N_CA_bond_length"], default_stds["N_CA_bond_length"], False))
    ca_c_blens_predictor = Sequential(MLP(latent_dim, 1, n_hidden=hidden_dim),
        ScaleShift(default_means["CA_C_bond_length"], default_stds["CA_C_bond_length"], False))
    c_n_blens_predictor = Sequential(MLP(latent_dim, 1, n_hidden=hidden_dim),
        ScaleShift(default_means["C_N_bond_length"], default_stds["C_N_bond_length"], False))

if settings['bins']['backbone_angle_bin']:
    n_ca_c_angle_predictor = MLP(latent_dim, settings['bins']['angle_bin_count'], n_hidden=hidden_dim)
    ca_c_n_angle_predictor = MLP(latent_dim, settings['bins']['angle_bin_count'], n_hidden=hidden_dim)
    c_n_ca_angle_predictor = MLP(latent_dim, settings['bins']['angle_bin_count'], n_hidden=hidden_dim)
else:
    n_ca_c_angle_predictor = Sequential(MLP(latent_dim, 1, n_hidden=hidden_dim),
        ScaleShift(default_means["N_CA_C_bond_angle"], default_stds["N_CA_C_bond_angle"], True))
    ca_c_n_angle_predictor = Sequential(MLP(latent_dim, 1, n_hidden=hidden_dim), 
        ScaleShift(default_means["CA_C_N_bond_angle"], default_stds["CA_C_N_bond_angle"], True))
    c_n_ca_angle_predictor = Sequential(MLP(latent_dim, 1, n_hidden=hidden_dim), 
        ScaleShift(default_means["C_N_CA_bond_angle"], default_stds["C_N_CA_bond_angle"], True))

sidechain_torsion_predictors = [MLP(latent_dim, settings['bins']['angle_bin_count'], n_hidden=hidden_dim) for _ in range(6)]
all_predictor_heads = [n_ca_c_angle_predictor, ca_c_n_angle_predictor, c_n_ca_angle_predictor] +\
     sidechain_torsion_predictors + [n_ca_blens_predictor, ca_c_blens_predictor, c_n_blens_predictor]



model = MultiHeadModel(rnn_encoder, all_predictor_heads)


# optimizer
optimizer = Adam(model.parameters(),
                 lr=settings['training']['lr'],
                 weight_decay=settings['training']['weight_decay'])



# loss
angle_digitizer = OnehotDigitizer((-179, 179, settings['bins']['angle_bin_count']), binary=False)

n_ca_blens_digitizer = OnehotDigitizer(settings['bins']['n-ca_bin'], binary=False)
ca_c_blens_digitizer = OnehotDigitizer(settings['bins']['ca-c_bin'], binary=False)
c_n_blens_digitizer = OnehotDigitizer(settings['bins']['c-n_bin'], binary=False)

loss_term_weights = settings['training']['loss_term_weigths']
if type(loss_term_weights) is not list:
    loss_term_weights = [loss_term_weights] * 12
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

    # angle_mask = (batch.angs != 0).float().to(device)
    # angle_targets = batch.angs * 180 / np.pi
    # angle_targets_labels = angle_digitizer.digitize(angle_targets).to(device)
    # angle_cross_entropy = cross_entropy(angle_pred_raw, angle_targets_labels)

    # masked_angle_loss = torch.sum(angle_cross_entropy * angle_mask, axis=-1)

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
        
        masked_blens_loss = torch.cat([n_ca_blens_loss, ca_c_blens_loss, c_n_blens_loss], dim=-1)
    else:
        blens_preds = torch.cat(preds[9:], axis=-1)
        blens_targets = batch.blens
        masked_blens_loss = numerical_loss(blens_preds, blens_targets, 
        [default_stds["N_CA_bond_length"], default_stds["CA_C_bond_length"], default_stds["C_N_bond_length"]])
    # # n-ca bond length losses
    # blens_mask = (batch.blens != 0).float().to(device)
    # n_ca_blens_targets_labels = n_ca_blens_digitizer.digitize(batch.blens[:, :, 0]).to(device)
    # n_ca_blens_pred_raw = torch.moveaxis(preds[12], -1, 1)
    # n_ca_blens_cross_entropy = cross_entropy(n_ca_blens_pred_raw, n_ca_blens_targets_labels)

    # # ca-c bond length losses
    # ca_c_blens_targets_labels = ca_c_blens_digitizer.digitize(batch.blens[:, :, 1]).to(device)
    # ca_c_blens_pred_raw = torch.moveaxis(preds[13], -1, 1)
    # ca_c_blens_cross_entropy = cross_entropy(ca_c_blens_pred_raw, ca_c_blens_targets_labels)

    # # c-n bond length losses
    # c_n_blens_targets_labels = c_n_blens_digitizer.digitize(batch.blens[:, :, 2]).to(device)
    # c_n_blens_pred_raw = torch.moveaxis(preds[14], -1, 1)
    # c_n_blens_cross_entropy = cross_entropy(c_n_blens_pred_raw, c_n_blens_targets_labels)

    # # bond length loss
    # blens_cross_entropy = torch.stack([n_ca_blens_cross_entropy, ca_c_blens_cross_entropy, c_n_blens_cross_entropy], axis=-1)
    # masked_blens_loss = torch.sum(blens_cross_entropy * blens_mask, axis=-1)

    # masked total loss
    mask = batch.msks.to(device)
    weighted_loss = torch.sum(torch.cat([masked_angle_loss, masked_blens_loss], dim=-1) * \
         torch.tensor(loss_term_weights).to(device), dim=-1)
    loss = torch.sum((weighted_loss * mask) / torch.sum(mask + 1e-8))

    # print('batch total loss:', loss.detach().cpu().numpy())

    return loss



# training
trainer = Trainer(
    model=model,
    loss_fn=compute_loss,
    optimizer=optimizer,
    device=device,
    yml_path=settings_path,
    output_path=settings['general']['output'],
    script_name=__file__,
    lr_scheduler=settings['training']['lr_scheduler'],
    bond_length_bin=settings['bins']['bond_length_bin'],
    backbone_angle_bin=settings['bins']['backbone_angle_bin'],
    bin_references={
        "angles": angle_digitizer.get_reference(),
        "n_ca_bond_length": n_ca_blens_digitizer.get_reference(),
        "ca_c_bond_length": ca_c_blens_digitizer.get_reference(),
        "c_n_bond_length": c_n_blens_digitizer.get_reference()
    },
    checkpoint_log=settings['checkpoint']['log'],
    checkpoint_val=settings['checkpoint']['val'],
    checkpoint_test=settings['checkpoint']['test'],
    checkpoint_model=settings['checkpoint']['model'],
    verbose=settings['checkpoint']['verbose'],
    preempt=settings['training']['preempt'],
    debug=debug_mode
)

trainer.print_layers()

best_results = trainer.train(
    epochs=settings['training']['epochs'],
    train_dataloader=train,
    val_dataloader=val,
    test_dataloader=test
)

# Log hyperparameters and the best performance of this run
run_hparams = settings['training']
run_hparams.update(settings['model'])
for hp in list(run_hparams):
    if type(run_hparams[hp]) in [bool, float, int]:
        run_hparams['hparams/'+hp] = run_hparams[hp]
    del run_hparams[hp]

hparam_metrics = {}
if best_results is not None:
    hparam_metrics["metrics/loss"] = best_results["val/loss"]
    hparam_metrics["metrics/rmse_bb_angle"] = best_results["val/rmse_bb_angle"]
    hparam_metrics["metrics/rmse_sc_tor"] = best_results["val/rmse_sc_tor"]
    hparam_metrics["metrics/rmse_blens"] = best_results["val/rmse_blens"]
trainer.logger.log_hparams(run_hparams, hparam_metrics)

print('done!')