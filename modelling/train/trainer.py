from copy import deepcopy
import os
import numpy as np
import pandas as pd
import torch
import time
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial

from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch import nn
from datetime import datetime
from modelling.train.metrics import rmse, structure_rmsd

from modelling.utils import Logger, max_sampling, tensor_to_numpy, set_model_parameters
from modelling.utils.plotting import plot_scatter

class Trainer:
    """
    Parameters
    ----------
    """
    def __init__(self,
                 model,
                 builder,
                 loss_fn,
                 optimizer,
                 device,
                 yml_path,
                 output_path,
                 script_name,
                 initial_lr,
                 lr_scheduler,
                 mode,
                 bin_references=None,
                 checkpoint_log=1,
                 checkpoint_val=1,
                 checkpoint_test=20,
                 checkpoint_model=1,
                 verbose=False,
                 training=True,
                 hooks=None,
                 save_test_batch=False,
                 preempt=False,
                 debug=False,
                 build_block_size=None,
                 central_residue=None):
        self.model = model
        self.builder = builder
        self.builder.set_prediction_conversion_func(partial(self.convert_prediction, data_format="tensor"))
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.bin_references = bin_references
        self.device = device
        self.preempt = preempt
        self.debug = debug
        self.build_block_size = build_block_size
        self.central_residue = central_residue
        self.current_lr = initial_lr

        if type(device) is list and len(device) > 1:
            self.multi_gpu = True
        else:
            self.multi_gpu = False
        self.verbose = verbose
        self.mode = mode


        # outputs
        last_checkpoint = self._subdirs(yml_path, output_path, script_name)
        
        if training:

            # hooks
            if hooks:
                self.hooks = None
                self._hooks(hooks)

            # learning rate scheduler
            self._handle_scheduler(lr_scheduler, optimizer)
            self.lr_scheduler = lr_scheduler

        # checkpoints
        self.check_log = checkpoint_log
        self.check_val = checkpoint_val
        self.check_test = checkpoint_test
        self.check_model = checkpoint_model

        # logging
        log_columns = ['epoch', 'lr', 'time', 'tr/loss', 'val/loss', 'test/loss',
                             'tr/rmse_bb_angle', 'val/rmse_bb_angle', 'test/rmse_bb_angle',
                             'tr/rmse_sc_tor', 'val/rmse_sc_tor', 'test/rmse_sc_tor',
                             'tr/rmse_blens', 'val/rmse_blens', 'test/rmse_blens']
        if mode == 'building':
            log_columns += ['tr/structure_rmsd', 'val/structure_rmsd', 'test/structure_rmsd']                             
        self.logger = Logger(self.output_path, log_columns)
        self.epoch = 0  # number of epochs of any steps that model has gone through so far
        self.rmsd_tracker = pd.DataFrame()

        self.best_val_loss = float("inf")
        self.save_test_batch = save_test_batch

        if preempt and last_checkpoint is not None:
            self.resume_model(last_checkpoint)

    def _handle_scheduler(self, lr_scheduler, optimizer):
        if lr_scheduler is None:
            self.scheduler = None


        elif lr_scheduler[0] == 'plateau':
            self.scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                               mode='min',
                                               patience=lr_scheduler[2],
                                               factor=lr_scheduler[3],
                                               min_lr=lr_scheduler[4])
        elif lr_scheduler[0] == 'decay':
            lambda1 = lambda epoch: np.exp(-epoch * lr_scheduler[1])
            self.scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda1, verbose=self.verbose)

        else:
            raise NotImplemented('scheduler "%s" is not implemented yet.'%lr_scheduler[0])

    def _subdirs(self, yml_path, output_path, script_name):
        last_checkpoint = None
        # create output directory and subdirectories
        path_iter = output_path[1]
        out_path = os.path.join(output_path[0], 'training_%i'%path_iter)
        while os.path.exists(out_path):
            path_iter+=1
            out_path = os.path.join(output_path[0],'training_%i'%path_iter)
        if self.preempt:
            # check whether can continue from last path_iter
            path_iter-=1
            parent_folder = os.path.join(output_path[0],'training_%i'%path_iter)
            if not os.path.exists(parent_folder) or 'preempt_lock' in os.listdir(parent_folder) \
                 or '1' not in os.listdir(parent_folder):
                # continue from last job not allowed or not possible
                path_iter+=1
                out_path = os.path.join(output_path[0],'training_%i'%path_iter, '1')
            else:
                # can continue last preempted job
                last_preempted = max([int(n) for n in os.listdir(parent_folder)])
                last_checkpoint = os.path.join(parent_folder, str(last_preempted), 'models/model_state.tar')
                out_path = os.path.join(output_path[0],'training_%i'%path_iter, str(last_preempted + 1))
                # make sure resume from a run that has saved model state
                while not os.path.exists(last_checkpoint):
                    last_preempted -= 1
                    if last_preempted == 0:
                        raise RuntimeError("Cannot find a checkpoint to resume from!")
                    last_checkpoint = os.path.join(parent_folder, str(last_preempted), 'models/model_state.tar')

        os.makedirs(out_path)
        self.output_path = out_path

        self.val_out_path = os.path.join(self.output_path, 'validation')
        os.makedirs(self.val_out_path)

        # subdir for computation graph
        self.graph_path = os.path.join(self.output_path, 'graph')
        if not os.path.exists(self.graph_path):
            os.makedirs(self.graph_path)

        # saved models
        self.model_path = os.path.join(self.output_path, 'models')
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        script_out = os.path.join(self.output_path, 'run_scripts')
        os.makedirs(script_out)
        shutil.copyfile(yml_path, os.path.join(script_out,os.path.basename(yml_path)))
        shutil.copyfile(script_name, os.path.join(script_out, os.path.basename(script_name)))

        if self.debug:
            self.debug_dir = os.path.join(self.output_path, 'debug')
            os.makedirs(self.debug_dir)
        return last_checkpoint

    def _hooks(self, hooks):
        hooks_list = []
        if 'vismolvector3d' in hooks and hooks['vismolvector3d']:
            from nmrpred.train.hooks import VizMolVectors3D

            vis = VizMolVectors3D()
            vis.set_output(True, None)
            hooks_list.append(vis)

        if len(hooks_list) > 0:
            self.hooks = hooks_list


    def print_layers(self):
        total_n_params = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)
                if len(param.shape) > 1:
                    total_n_params += param.shape[0] * param.shape[1]
                else:
                    total_n_params += param.shape[0]
        print('\n total trainable parameters: %i\n' % total_n_params)

    def plot_grad_flow(self):
        ave_grads = []
        layers = []
        for n, p in self.model.named_parameters():
            if (p.requires_grad) and ("bias" not in n):
                # shorten names
                layer_name = n.split('.')
                layer_name = [l[:3] for l in layer_name]
                layer_name = '.'.join(layer_name[:-1])
                layers.append(layer_name)
                # print(layer_name, p.grad)
                if p.grad is not None:
                    ave_grads.append(p.grad.abs().mean().detach().cpu())
                else:
                    ave_grads.append(0)

        fig, ax = plt.subplots(1, 1)
        ax.plot(ave_grads, alpha=0.3, color="b")
        ax.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow: epoch#%i" %self.epoch)
        plt.grid(True)
        ax.set_axisbelow(True)

        file_name= os.path.join(self.graph_path,"avg_grad.png")
        plt.savefig(file_name, dpi=300,bbox_inches='tight')
        plt.close(fig)


    def _optimizer_to_device(self):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device[0])

    

    


    def resume_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        if checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Resumed training from checkpoint [%s]" % path)
        return loss


    def validate(self, data_loader, use_builder=True, central_residue=None):
        self.model.eval()
        self.model.requires_dr = False

        losses = []
        angle_preds = []
        blens_preds = []
        sc_blens_preds = []
        sc_ang_preds = []
        angle_targets = []
        blens_targets = []
        sc_blens_targets = []
        sc_ang_targets = []
        structure_rmsds = []
        labels = []

        for batch in tqdm(data_loader):
            with torch.no_grad():
                results = self.predict_and_evaulate(batch, use_builder=use_builder, central_residue=central_residue)
            losses.append(tensor_to_numpy(results['loss']))
            if central_residue is None:
                angle_preds.extend(results['angle_preds'])
                blens_preds.extend(results['blens_preds'])
                sc_blens_preds.extend(results['sc_blens_preds'])
                sc_ang_preds.extend(results['sc_ang_preds'])
                angle_targets.extend(results['angle_targets'])
                blens_targets.extend(results['blens_targets'])
                sc_blens_targets.extend(results['sc_blens_targets'])
                sc_ang_targets.extend(results['sc_ang_targets'])
            else:
                angle_preds.append(results['angle_preds'][:, central_residue])
                blens_preds.append(results['blens_preds'][:, central_residue])
                angle_targets.append(results['angle_targets'][:, central_residue])
                blens_targets.append(results['blens_targets'][:, central_residue])
            if 'structure_rmsds' in results:
                structure_rmsds.extend(results['structure_rmsds'])
            labels.extend(results["ids"])

        angle_preds = np.concatenate(angle_preds, axis=0)
        blens_preds = np.concatenate(blens_preds)
        sc_blens_preds = np.concatenate(sc_blens_preds)
        sc_ang_preds = np.concatenate(sc_ang_preds)
        angle_targets = np.concatenate(angle_targets)
        blens_targets = np.concatenate(blens_targets)
        sc_blens_targets = np.concatenate(sc_blens_targets)
        sc_ang_targets = np.concatenate(sc_ang_targets)

        bb_angle_rmse = rmse(angle_preds[:,:3], angle_targets[:,:3] * 180 / np.pi, periodic=True)
        sc_tor_rmse = rmse(angle_preds[:,3:], angle_targets[:,3:] * 180 / np.pi, periodic=True)
        blens_rmse = rmse(blens_preds, blens_targets)
        sc_blens_rmse = rmse(sc_blens_preds, sc_blens_targets)
        sc_ang_rmse = rmse(sc_ang_preds, sc_ang_targets * 180 / np.pi, periodic=True)


        outputs = dict()
        outputs['loss'] = np.mean(losses)
        outputs['angle_preds'] = angle_preds
        outputs['blens_preds'] = blens_preds
        outputs['sc_blens_preds'] = sc_blens_preds
        outputs['sc_ang_preds'] = sc_ang_preds
        outputs['angle_targets'] = angle_targets
        outputs["blens_targets"] = blens_targets
        outputs['sc_blens_targets'] = sc_blens_targets
        outputs['sc_ang_targets'] = sc_ang_targets
        outputs["bb_angle_rmse"] = bb_angle_rmse
        outputs["sc_tor_rmse"] = sc_tor_rmse
        outputs["blens_rmse"] = blens_rmse
        outputs['sc_blens_rmse'] = sc_blens_rmse
        outputs['sc_ang_rmse'] = sc_ang_rmse
        outputs["labels"] = labels

        outputs['structure_rmsds'] = structure_rmsds
        outputs['mean_structure_rmsd'] = np.mean(structure_rmsds)
        return outputs

    def convert_prediction(self, preds, data_format="numpy"):

        if data_format == "numpy":
            angle_preds = tensor_to_numpy(torch.cat([preds["theta1"], preds["theta2"], preds["theta3"]], axis=-1)) * 180/np.pi
            if "chis" in preds:
                sidechain_torsion_preds = max_sampling(torch.stack(preds["chis"], axis=2),
                                        self.bin_references["angles"],
                                        "numpy")
            else:
                sidechain_torsion_preds = np.concatenate([np.zeros_like(angle_preds)] * 2, axis=-1)
            angle_preds = np.concatenate([angle_preds, sidechain_torsion_preds], axis=-1)
        elif data_format == "tensor":
            angle_preds = torch.cat([preds["theta1"], preds["theta2"], preds["theta3"]], axis=-1) * 180/np.pi
            if "chis" in preds:
                sidechain_torsion_preds = max_sampling(torch.stack(preds["chis"], axis=2),
                                    self.bin_references["angles"],
                                    "tensor")
            else:
                sidechain_torsion_preds = torch.cat([torch.zeros_like(angle_preds)] * 2, axis=-1)
            angle_preds = torch.cat([angle_preds, sidechain_torsion_preds], axis=-1)


        if data_format == "numpy":
            blens_preds = tensor_to_numpy(torch.cat([preds["d1"], preds["d2"], preds["d3"]], axis=-1))
        elif data_format == "tensor":
            blens_preds = torch.cat([preds["d1"], preds["d2"], preds["d3"]], axis=-1)

        # sidechain bond lengths and bond angles
        if data_format == "numpy":
            if "r1" in preds:
                sc_blens_preds = tensor_to_numpy(preds["r1"])
            else:
                sc_blens_preds = np.zeros(angle_preds.shape[:-1] + (1,))
            if "alpha1" in preds:
                sc_ang_preds = tensor_to_numpy(preds["alpha1"]) * 180/np.pi
            else:
                sc_ang_preds = np.zeros(angle_preds.shape[:-1] + (1,))
        elif data_format == "tensor":
            if "r1" in preds:
                sc_blens_preds = preds["r1"]
            else:
                sc_blens_preds = torch.zeros(angle_preds.shape[:-1] + (1,), device=angle_preds.device)
            if "alpha1" in preds:
                sc_ang_preds = preds["alpha1"] * 180/np.pi
            else:
                sc_ang_preds = torch.zeros(angle_preds.shape[:-1] + (1,), device=angle_preds.device)
        return angle_preds, blens_preds, sc_blens_preds, sc_ang_preds

    def log_graph(self, save_model, batch):
        batch_inputs = {"phi": batch.angs[:, :, 0].to(self.device[0]),
                        "psi": batch.angs[:, :, 1].to(self.device[0]),
                        "omega": batch.angs[:, :, 2].to(self.device[0]),
                        "chi1": batch.angs[:, :, 6].to(self.device[0]),
                        "res_type": torch.argmax(batch.seqs, axis=-1).to(self.device[0]),
                        "lengths": torch.tensor(batch.lengths)}
        self.logger.log_graph(save_model, batch_inputs)

    def predict_and_evaulate(self, batch, use_builder=False, build_by_block=None, central_residue=None, sample_weights=None):
        batch_inputs = {"phi": batch.angs[:, :, 0].to(self.device[0]),
                        "psi": batch.angs[:, :, 1].to(self.device[0]),
                        "omega": batch.angs[:, :, 2].to(self.device[0]),
                        "chi1": batch.angs[:, :, 6].to(self.device[0]),
                        "res_type": torch.argmax(batch.seqs, axis=-1).to(self.device[0]),
                        "lengths": batch.lengths}
        build_range = None
        if build_by_block is not None:
            build_start = [np.random.randint(0, max(l - build_by_block, 1)) for l in batch.lengths]
            build_range = [(start, min(start + build_by_block, l)) for start, l in zip(build_start, batch.lengths)]
            batch_inputs.update({"build_range": build_range})

        if use_builder:
            preds = self.builder(batch_inputs)
            pred_dmats = preds["dist_mats"]
            pred_coords = preds["coords"]
            preds = preds["predictions"]
        else:
            preds = self.model(batch_inputs)

        if "scalar" in self.mode:
            if sample_weights is not None:
                loss = self.loss_fn(preds, batch, sample_weights=sample_weights)
            else:
                loss = self.loss_fn(preds, batch)
            
        elif "building" in self.mode:
            loss = self.loss_fn(pred_dmats, batch, build_range)

        angle_preds, blens_preds, sc_blens_preds, sc_ang_preds = self.convert_prediction(preds)
        if central_residue is not None:
            backbone_angle_rmse = rmse(angle_preds[:, central_residue, :3], batch.angs[:, central_residue, 3:6] * 180 / np.pi, periodic=True)
            sidechain_torsion_rmse = rmse(angle_preds[:, central_residue, 3:], batch.angs[:, central_residue, 6:] * 180 / np.pi, periodic=True)
            blens_rmse = rmse(blens_preds[:, central_residue], batch.blens[:, central_residue])
        else:
            backbone_angle_rmse = rmse(angle_preds[:, :, :3], batch.angs[:, :, 3:6] * 180 / np.pi, periodic=True)
            sidechain_torsion_rmse = rmse(angle_preds[:, :, 3:], batch.angs[:, :, 6:] * 180 / np.pi, periodic=True)
            blens_rmse = rmse(blens_preds, batch.blens)
            sidechain_blens_rmse = rmse(sc_blens_preds, batch.sc_blens)
            sidechain_ang_rmse = rmse(sc_ang_preds, batch.sc_angs * 180 / np.pi, periodic=True)

        results = {"loss": loss,
                   "angle_preds": angle_preds,
                   "blens_preds": blens_preds,
                   "sc_blens_preds": sc_blens_preds,
                   "sc_ang_preds": sc_ang_preds,
                   "angle_targets": tensor_to_numpy(batch.angs[:,:,3:]),
                   "blens_targets": tensor_to_numpy(batch.blens),
                   "sc_blens_targets": tensor_to_numpy(batch.sc_blens),
                   "sc_ang_targets": tensor_to_numpy(batch.sc_angs),
                   "backbone_angle_rmse": backbone_angle_rmse,
                   "sidechain_torsion_rmse": sidechain_torsion_rmse,
                   "blens_rmse": blens_rmse,
                   "sc_blens_rmse": sidechain_blens_rmse,
                   "sc_ang_rmse": sidechain_ang_rmse,
                   "ids": batch.pids}

        if use_builder:
            structure_rmsds = structure_rmsd(pred_coords, batch.crds, batch.lengths, False, build_range)
            structure_rmsd_mean = np.mean(structure_rmsds)
            results.update({
                "structure_rmsds": structure_rmsds,
                "mean_structure_rmsd": structure_rmsd_mean
            })
        
        return results

    def get_resampled_weights(self, batch_train, batch_val):
        """
        Using the resample technique (Learning to Reweight Examples for Robust Deep Learning, 
        https://arxiv.org/pdf/1803.09050.pdf) to train the model
        """
        batch_inputs_train = {"phi": batch_train.angs[:, :, 0].to(self.device[0]),
                            "psi": batch_train.angs[:, :, 1].to(self.device[0]),
                            "omega": batch_train.angs[:, :, 2].to(self.device[0]),
                            "chi1": batch.angs[:, :, 6].to(self.device[0]),
                            "res_type": torch.argmax(batch_train.seqs, axis=-1).to(self.device[0]),
                            "lengths": batch_train.lengths}
        pred_model = deepcopy(self.model)
        dual_model = deepcopy(self.model)
        preds_train = pred_model(batch_inputs_train)
        sample_weights = torch.zeros(batch_train.msks.shape, requires_grad=True, dtype=torch.float).to(self.device[0])
        train_loss = self.loss_fn(preds_train, batch_train, sample_weights=sample_weights)
        # train_loss = torch.mean(sample_weights.flatten())

        # # save model state
        # model_state = deepcopy(self.model.state_dict())

        # first backward pass with retaining gradients with respect to sample weights
        # self.model.zero_grad()
        # train_loss.backward(create_graph=True)
        # gradient_containing_params = {k:v for k,v in pred_model.named_parameters()}
        # # do one "proposed" gradient update step and evaluate on validation data
        # updated_state_dict = {k: gradient_containing_params[k].detach() - self.current_lr * gradient_containing_params[k].grad 
        #     for k in gradient_containing_params}


        params = {k:v for k,v in pred_model.named_parameters()}
        param_grads = torch.autograd.grad(train_loss, params.values(), create_graph=True)
        param_grads = dict(zip(params.keys(), param_grads))
        updated_state_dict = {k: params[k].detach() - self.current_lr * param_grads[k] for k in params}
        # for k in current_state_dict:
        #     if k in gradient_containing_params:
        #         updated_state_dict[k] = current_state_dict[k] - gradient_containing_params[k].grad
        #     else:
        #         updated_state_dict[k] = current_state_dict[k]

        set_model_parameters(dual_model, updated_state_dict)
        # scalar_prediction_model.load_state_dict(updated_state_dict)

        batch_inputs_val = {"phi": batch_val.angs[:, :, 0].to(self.device[0]),
                            "psi": batch_val.angs[:, :, 1].to(self.device[0]),
                            "omega": batch_val.angs[:, :, 2].to(self.device[0]),
                            "res_type": torch.argmax(batch_val.seqs, axis=-1).to(self.device[0]),
                            "lengths": batch_val.lengths}


        preds_val = dual_model(batch_inputs_val)
        val_loss = self.loss_fn(preds_val, batch_val)
        # val_loss = torch.mean(torch.cat([item.flatten() for item in preds_val]))
        sample_weight_grads = torch.autograd.grad(val_loss, sample_weights)[0].detach()
        # sample_weight_grads = torch.zeros_like(sample_weights)

        # calculate adjusted sample weights from gradients
        clamped_grads = torch.maximum(-sample_weight_grads, torch.tensor(0., device=sample_weight_grads.device))
        grads_sum = torch.sum(clamped_grads) + 1e-8
        new_sample_weights = clamped_grads / grads_sum

        # # recover old model state
        # self.model.load_state_dict(model_state)
        # self.model.zero_grad()

        # delete models
        # empty_state_dict = {k: None for k in gradient_containing_params}
        # set_model_parameters(pred_model, empty_state_dict)
        # set_model_parameters(dual_model, empty_state_dict)
        for p in pred_model.parameters():
            p.grad = None   
        torch.cuda.empty_cache()
        return new_sample_weights


        

    def train(self,
              epochs,
              train_dataloader,
              infinite_val_data_loader=None,
              val_dataloader=None,
              test_dataloader=None,
              clip_grad=0):
        """
        The main function to train model for the given number of epochs (and steps per epochs).
        The implementation allows for resuming the training with different data and number of epochs.

        Parameters
        ----------
        epochs: int
            number of training epochs

        steps: int
            number of steps to call it an epoch (designed for nonstop data generators)


        """
        self.model.to(self.device[0])
        self.builder.to(self.device[0])
        save_model = self.model
        if self.multi_gpu:
            self.model = nn.DataParallel(self.model, device_ids=self.device)
        self._optimizer_to_device()

        running_val_loss = []
        last_test_epoch = -100
        best_epoch_results = None

        has_logged_graph = False

        # evaluate test performance prior to training when building model
        if test_dataloader is not None and 'building' in self.mode:
            test_results = self.validate(test_dataloader, use_builder="building" in self.mode, central_residue=self.central_residue)
            test_struc_rmsd = test_results['mean_structure_rmsd']
            iteration_rmsds = {'epoch': self.epoch}
            iteration_rmsds.update({pdb_name: rmsd for pdb_name, rmsd in zip(test_results["labels"], test_results['structure_rmsds'])})
            self.rmsd_tracker = self.rmsd_tracker.append(iteration_rmsds, ignore_index=True)
            self.rmsd_tracker.to_csv(os.path.join(self.output_path, "test_protein_rmsds.csv"))


        while 1:
            t0 = time.time()

            # record total number of epochs so far
            self.epoch += 1
            if self.epoch > epochs:
                break


            # training
            running_loss = 0.0
            rmse_bb_angles = []
            rmse_sc_tor = []
            rmse_bonds = []
            rmse_sc_blens = []
            rmse_sc_angs = []
            structure_rmsds = []
            step_losses = []
            step_labels = []
            self.model.train()
            self.optimizer.zero_grad()

            if not self.verbose:
                train_dataloader = tqdm(train_dataloader)
            
            s = 1
            last_state_dict = None
            last_optimizer_state_dict = None
            last_batch = second_last_batch = None
            for batch in train_dataloader:
                if not has_logged_graph:
                    # self.log_graph(save_model, batch)
                    has_logged_graph = True
                self.optimizer.zero_grad()
                second_last_state_dict = last_state_dict
                second_last_optimizer_state_dict = last_optimizer_state_dict
                last_state_dict = deepcopy(self.model.state_dict())
                last_optimizer_state_dict = deepcopy(self.optimizer.state_dict())
                if "scalar" in self.mode:
                    if infinite_val_data_loader is None:
                        result = self.predict_and_evaulate(batch, use_builder=False, central_residue=self.central_residue)
                    else:
                        val_for_sample_weights = next(infinite_val_data_loader)
                        
                        # double backpropagation for RNN is not supported by cudnn, so need to disable cudnn here
                        with torch.backends.cudnn.flags(enabled=False):
                            sample_weights = self.get_resampled_weights(batch, val_for_sample_weights)
                        self.optimizer.zero_grad()
                        result = self.predict_and_evaulate(batch, use_builder=False, central_residue=self.central_residue, 
                        sample_weights=sample_weights)
                        # result = self.predict_and_evaulate(batch, use_builder=False, central_residue=self.central_residue, 
                        # sample_weights=None)
                elif "building" in self.mode:
                    result = self.predict_and_evaulate(batch, use_builder=True, build_by_block=self.build_block_size, central_residue=self.central_residue)
                loss = result['loss']
                loss.backward()
                if clip_grad>0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                self.optimizer.step()
                # if (s+1)%4==0 or s==steps-1:
                #     self.optimizer.step()          # if in, comment out the one in the loop
                #     self.optimizer.zero_grad()     # if in, comment out the one in the loop

                current_loss = loss.detach().item()
                running_loss += current_loss

                # calculate training RMSE
                
                rmse_bb_angles.append(result['backbone_angle_rmse'])
                rmse_sc_tor.append(result['sidechain_torsion_rmse'])
                rmse_bonds.append(result['blens_rmse'])
                rmse_sc_blens.append(result['sc_blens_rmse'])
                rmse_sc_angs.append(result['sc_ang_rmse'])
                if 'mean_structure_rmsd' in result:
                    structure_rmsds.append(result['mean_structure_rmsd'])
                if current_loss > 100:
                    if self.debug:
                        current_model = self.model
                        current_optimizer = self.optimizer
                        current_batch_data = batch
                        torch.save({
                            "second_last_model_state": second_last_state_dict,
                            "last_model_state": last_state_dict,
                            "model": current_model,
                            "second_last_optimizer_state": second_last_optimizer_state_dict,
                            "last_optimizer_state": last_optimizer_state_dict,
                            "optimizer": current_optimizer,
                            "batch": current_batch_data,
                            "last_batch": last_batch,
                            "second_last_batch": second_last_batch
                        }, os.path.join(self.debug_dir, "loss_explosion.pkl"))
                    else:
                        # return best_epoch_results, False
                        print("Warning! Large loss occured:", current_loss)

                if self.debug:
                    step_losses.append(current_loss)
                    step_labels.append(result['ids'])
                    

                second_last_batch = last_batch
                last_batch = deepcopy(batch)

                if self.verbose:
                    print(datetime.now(),
                        "Train: Epoch %i/%i - step %i - loss: %.5f - running_loss: %.5f"
                        % (self.epoch, epochs, s, current_loss, running_loss / s))

                
                s += 1

            running_loss /= (s-1)

            rmse_bb_angles = np.mean(rmse_bb_angles[-100:])
            rmse_sc_tor = np.mean(rmse_sc_tor[-100:])
            rmse_bonds = np.mean(rmse_bonds[-100:])
            rmse_sc_blens = np.mean(rmse_sc_blens[-100:])
            rmse_sc_angs = np.mean(rmse_sc_angs[-100:])
            if len(structure_rmsds) > 0:
                structure_rmsds = np.mean(structure_rmsds[-100:])
            else:
                structure_rmsds = 0

            # save debug info when necessary
            if self.debug:
                torch.save({"step_losses": step_losses,
                "step_ids": step_labels}, os.path.join(self.debug_dir, str(self.epoch) + ".pkl"))

            # plots
            self.plot_grad_flow()

            # validation
            val_error = float("inf")

            val_bb_angle_rmse = np.nan
            val_sc_tor_rmse = np.nan
            val_blens_rmse = np.nan
            val_sc_blens_rmse = np.nan
            val_sc_ang_rmse = np.nan
            val_struc_rmsd = np.nan
            val_struc_all_rmsds = []
            if val_dataloader is not None and \
                self.epoch % self.check_val == 0:
                if self.verbose:
                    print("Start validation")
                val_results = self.validate(val_dataloader, use_builder="building" in self.mode, central_residue=self.central_residue)
                val_error = val_results["loss"]
                val_bb_angle_rmse = val_results['bb_angle_rmse']
                val_sc_tor_rmse = val_results['sc_tor_rmse']
                val_blens_rmse = val_results['blens_rmse']
                val_sc_blens_rmse = val_results['sc_blens_rmse']
                val_sc_ang_rmse = val_results['sc_ang_rmse']
                val_struc_rmsd = val_results['mean_structure_rmsd']
                val_struc_all_rmsds = val_results['structure_rmsds']

                # log RMSD distributions
                if len(val_struc_all_rmsds) > 0:
                    self.logger.log_hist(val_struc_all_rmsds, "validation/structure_rmsds", self.epoch)


            # checkpoint every epoch when preempt is allowed
            if self.preempt:
                state_dict = {
                            'epoch': self.epoch,
                            'model_state_dict': save_model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': loss
                            }
                if self.lr_scheduler is not None:
                    state_dict['scheduler_state_dict'] = self.scheduler.state_dict()
                else:
                    state_dict['scheduler_state_dict'] = None
                torch.save(state_dict,
                    os.path.join(self.model_path, 'model_state.tar'))

            # checkpoint model when validation error gets lower
            test_error = np.nan
            test_bb_angle_rmse = np.nan
            test_sc_tor_rmse = np.nan
            test_blens_rmse = np.nan
            test_sc_blens_rmse = np.nan
            test_sc_ang_rmse = np.nan
            test_struc_rmsd = np.nan
            new_best = False
            if self.best_val_loss > val_error:
                new_best = True
                self.best_val_loss = val_error

                torch.save(save_model,#.state_dict(),
                           os.path.join(self.model_path, 'best_model.pt'))
                state_dict = {
                            'epoch': self.epoch,
                            'model_state_dict': save_model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': loss
                            }
                if self.lr_scheduler is not None:
                    state_dict['scheduler_state_dict'] = self.scheduler.state_dict()
                else:
                    state_dict['scheduler_state_dict'] = None
                torch.save(state_dict,
                    os.path.join(self.model_path, 'best_model_state.tar')
                )

                

                # save test predictions
                if test_dataloader is not None and self.epoch - last_test_epoch >= self.check_test:
                    if self.verbose:
                        print("Start testing")
                    test_results = self.validate(test_dataloader, use_builder="building" in self.mode, central_residue=self.central_residue)
                    test_error = test_results['loss']
                    test_bb_angle_rmse = test_results['bb_angle_rmse']
                    test_sc_tor_rmse = test_results['sc_tor_rmse']
                    test_blens_rmse = test_results['blens_rmse']
                    test_sc_blens_rmse = test_results['sc_blens_rmse']
                    test_sc_ang_rmse = test_results['sc_ang_rmse']
                    torch.save(test_results, os.path.join(self.val_out_path, 'test_results.pkl'))
                    
                    # save scatter plot
                    bb_angle_plot = plot_scatter(test_results['angle_preds'][:,:3], 
                                              test_results['angle_targets'][:,:3] * 180 / np.pi,
                                              "predictions",
                                              "targets",
                                              "backbone angles")
                    self.logger.log_figure(bb_angle_plot, "backbone_angle_corr", self.epoch, "test")
                    sc_tor_plot = plot_scatter(test_results['angle_preds'][:,3:], 
                                              test_results['angle_targets'][:,3:] * 180 / np.pi,
                                              "predictions",
                                              "targets",
                                              "sidechain torsions")
                    self.logger.log_figure(sc_tor_plot, "sidechain_torsion_corr", self.epoch, "test")
                    blens_plot = plot_scatter(test_results['blens_preds'],
                                              test_results['blens_targets'],
                                              "predictions",
                                              "targets",
                                              "bond lengths")
                    self.logger.log_figure(blens_plot, "blens_corr", self.epoch, "test")
                    sc_blens_plot = plot_scatter(test_results['sc_blens_preds'],
                                                 test_results['sc_blens_targets'],
                                                 "predictions",
                                                 "targets",
                                                 "sidechain bond lengths")
                    self.logger.log_figure(sc_blens_plot, "sc_blens_corr", self.epoch, "test")
                    sc_ang_plot = plot_scatter(test_results['sc_ang_preds'],
                                                test_results['sc_ang_targets'],
                                                "predictions",
                                                "targets",
                                                "sidechain bond angles")
                    self.logger.log_figure(sc_ang_plot, "sc_ang_corr", self.epoch, "test")
                    last_test_epoch = self.epoch
                    # np.save(os.path.join(self.val_out_path, 'test_Ei_epoch%i'%self.epoch), outputs['Ei'])

                    # save rmsd tracking
                    if  'building' in self.mode:
                        test_struc_rmsd = test_results['mean_structure_rmsd']
                        iteration_rmsds = {'epoch': self.epoch}
                        iteration_rmsds.update({pdb_name: rmsd for pdb_name, rmsd in zip(test_results["labels"], test_results['structure_rmsds'])})
                        self.rmsd_tracker = self.rmsd_tracker.append(iteration_rmsds, ignore_index=True)
                        self.rmsd_tracker.to_csv(os.path.join(self.output_path, "test_protein_rmsds.csv"))

            # learning rate decay
            if self.lr_scheduler is not None:
                if self.lr_scheduler[0] == 'plateau':
                    running_val_loss.append(val_error)
                    if len(running_val_loss) > self.lr_scheduler[1]:
                        running_val_loss.pop(0)
                    accum_val_loss = np.mean(running_val_loss)
                    self.scheduler.step(accum_val_loss)
                elif self.lr_scheduler[0] == 'decay':
                    self.scheduler.step()
                    accum_val_loss = 0.0

            # logging and best result handling
            for i, param_group in enumerate(
                        self.optimizer.param_groups):
                        # self.scheduler.optimizer.param_groups):
                    old_lr = float(param_group["lr"])
            self.current_lr = old_lr
            
            epoch_results = {
                "epoch": self.epoch,
                "lr": old_lr,
                "time": time.time() - t0,
                "tr/loss": running_loss,
                "val/loss": val_error,
                "test/loss": test_error,
                "tr/rmse_bb_angle": rmse_bb_angles,
                'val/rmse_bb_angle': val_bb_angle_rmse,
                'test/rmse_bb_angle': test_bb_angle_rmse,
                "tr/rmse_sc_tor": rmse_sc_tor,
                'val/rmse_sc_tor': val_sc_tor_rmse,
                'test/rmse_sc_tor': test_sc_tor_rmse,
                'tr/rmse_blens': rmse_bonds,
                'val/rmse_blens': val_blens_rmse,
                'test/rmse_blens': test_blens_rmse,
                "tr/rmse_sc_blens": rmse_sc_blens,
                "val/rmse_sc_blens": val_sc_blens_rmse,
                "test/rmse_sc_blens": test_sc_blens_rmse,
                "tr/rmse_sc_angs": rmse_sc_angs,
                "val/rmse_sc_angs": val_sc_ang_rmse,
                "test/rmse_sc_angs": test_sc_ang_rmse
            }

            if 'building' in self.mode:
                epoch_results.update({
                    'tr/structure_rmsd': structure_rmsds,
                    'val/structure_rmsd': val_struc_rmsd,
                    'test/structure_rmsd': test_struc_rmsd
                })

            if new_best:
                best_epoch_results = epoch_results

            if self.epoch % self.check_log == 0:
                self.logger.log_result(epoch_results)
        return best_epoch_results, True

    def log_statistics(self, data_collection):
        with open(os.path.join(self.output_path, "stats.txt"), "w") as f:
            f.write("Train data: %d\n" % data_collection.train[2])
            f.write("Val data: %d\n" % data_collection.val[2])
            f.write("Test data: %d\n" % data_collection.test[2])
            f.write("Normalizer: %s\n" % str(data_collection.normalizer))
            f.write("Test target hash: %s\n" % data_collection.hash)
