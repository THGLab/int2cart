import yaml
import sys
import torch
from torch import nn
from torch.optim import Adam, SGD
import numpy as np

from modelling.models.builder import BackboneBuilder
from modelling.utils import OnehotDigitizer
from sidechainnet import load
from modelling.train import Trainer
from modelling.utils.default_scalers import *
from modelling.utils.get_gpu import handle_gpu
from modelling.losses.scalar_losses import prepare_losses
from modelling.utils.infinite_data_loader import InfiniteDataLoader

from modelling.utils.load_data import load_data

torch.autograd.set_detect_anomaly(False)

#
settings_path = '../configs/debug.yml'
debug_mode = True
if len(sys.argv) > 1:
    settings_path = sys.argv[-1]
    debug_mode = False
settings = yaml.safe_load(open(settings_path, "r"))

device = [torch.device(dev) for dev in handle_gpu(settings['general']['device'])]


# data
train, val, test, infinite_val = load_data(settings)

# model
# model = get_model(settings)
builder = BackboneBuilder(settings)
model = builder.predictor

if 'pretrained_state' in settings['training']:
    model_state = torch.load(settings['training']['pretrained_state'])["model_state_dict"]
    builder.load_predictor_weights(model_state)


# optimizer
optimizer = settings['training'].get('optimizer', "Adam")
if optimizer.lower() == 'adam':
    optimizer = Adam(model.parameters(),
                    lr=settings['training']['lr'],
                    weight_decay=settings['training']['weight_decay'])
elif optimizer.lower() == 'sgd':
    momentum = settings['training'].get('momentum', 0)
    nestrov = settings['training'].get('nestrov', False)
    optimizer = SGD(model.parameters(),
                    lr=settings['training']['lr'],
                    weight_decay=settings['training']['weight_decay'],
                    momentum=momentum,
                    nesterov=nestrov)




# loss
angle_digitizer = OnehotDigitizer((-179, 179, settings['bins']['angle_bin_count']), binary=False)

n_ca_blens_digitizer = OnehotDigitizer(settings['bins']['n-ca_bin'], binary=False)
ca_c_blens_digitizer = OnehotDigitizer(settings['bins']['ca-c_bin'], binary=False)
c_n_blens_digitizer = OnehotDigitizer(settings['bins']['c-n_bin'], binary=False)

loss_fn = prepare_losses(settings, 
                         angle_digitizer, 
                         n_ca_blens_digitizer,
                         ca_c_blens_digitizer,
                         c_n_blens_digitizer,
                         rescale_by_length=settings['training'].get('rescale_loss_by_lengths', False))


# training
trainer = Trainer(
    model=model,
    builder=builder,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device=device,
    yml_path=settings_path,
    output_path=settings['general']['output'],
    script_name=__file__,
    initial_lr=settings['training']['lr'],
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
    debug=debug_mode,
    mode="scalar+building"
)

trainer.print_layers()

best_results, succeeded = trainer.train(
    epochs=settings['training']['epochs'],
    train_dataloader=train,
    val_dataloader=val,
    test_dataloader=test,
    infinite_val_data_loader=infinite_val
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
    hparam_metrics["metrics/structure_rmsd"] = best_results["val/structure_rmsd"]
trainer.logger.log_hparams(run_hparams, hparam_metrics)

if succeeded:
    print('Done!')
else:
    print('Error during training!')