import os
import numpy as np
import pandas as pd
import torch
import time
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch import nn
from datetime import datetime
from modelling.train.metrics import rmse

from modelling.utils import Logger, max_sampling, tensor_to_numpy
from modelling.utils.plotting import plot_scatter

class Trainer:
    """
    Parameters
    ----------
    """
    def __init__(self,
                 model,
                 loss_fn,
                 optimizer,
                 device,
                 yml_path,
                 output_path,
                 script_name,
                 lr_scheduler,
                 bond_length_bin,
                 backbone_angle_bin,
                 bin_references=None,
                 checkpoint_log=1,
                 checkpoint_val=1,
                 checkpoint_test=20,
                 checkpoint_model=1,
                 verbose=False,
                 training=True,
                 hooks=None,
                 save_test_batch=False,
                 preempt=False):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.bond_length_bin = bond_length_bin
        self.backbone_angle_bin = backbone_angle_bin
        self.bin_references = bin_references
        self.device = device
        self.preempt = preempt

        if type(device) is list and len(device) > 1:
            self.multi_gpu = True
        else:
            self.multi_gpu = False
        self.verbose = verbose


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
        self.logger = Logger(self.output_path, 
                            ['epoch', 'lr', 'time', 'loss', 'val_loss', 'test_loss',
                             'tr_err_bb_angle(RMSE)', 'val_err_bb_angle(RMSE)', 'test_err_bb_angle(RMSE)',
                             'tr_err_sc_tor(RMSE)', 'val_err_sc_tor(RMSE)', 'test_err_sc_tor(RMSE)',
                             'tr_err_blens(RMSE)', 'val_err_blens(RMSE)', 'test_err_blens(RMSE)'])
        self.epoch = 0  # number of epochs of any steps that model has gone through so far

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


    def validate(self, data_loader):
        self.model.eval()
        self.model.requires_dr = False

        losses = []
        angle_preds = []
        blens_preds = []
        angle_targets = []
        blens_targets = []
        labels = []

        for batch in tqdm(data_loader):
            with torch.no_grad():
                results = self.predict_and_evaulate(batch)
            losses.append(tensor_to_numpy(results['loss']))
            angle_preds.extend(results['angle_preds'])
            blens_preds.extend(results['blens_preds'])
            angle_targets.extend(results['angle_targets'])
            blens_targets.extend(results['blens_targets'])
            labels.extend(results["ids"])

        angle_preds = np.concatenate(angle_preds, axis=0)
        blens_preds = np.concatenate(blens_preds)
        angle_targets = np.concatenate(angle_targets)
        blens_targets = np.concatenate(blens_targets)

        bb_angle_rmse = rmse(angle_preds[:,:3], angle_targets[:,:3] * 180 / np.pi, periodic=True)
        sc_tor_rmse = rmse(angle_preds[:,3:], angle_targets[:,3:] * 180 / np.pi, periodic=True)
        blens_rmse = rmse(blens_preds, blens_targets)


        outputs = dict()
        outputs['loss'] = np.mean(losses)
        outputs['angle_preds'] = angle_preds
        outputs['blens_preds'] = blens_preds
        outputs['angle_targets'] = angle_targets
        outputs["blens_targets"] = blens_targets
        outputs["bb_angle_rmse"] = bb_angle_rmse
        outputs["sc_tor_rmse"] = sc_tor_rmse
        outputs["blens_rmse"] = blens_rmse
        outputs["labels"] = labels
        return outputs

    def convert_prediction(self, preds):
        if self.backbone_angle_bin:
            angle_preds = max_sampling(torch.stack(preds[:9], axis=-1),
                                    self.bin_references["angles"])
        else:
            backbone_angle_preds = tensor_to_numpy(torch.cat(preds[:3], axis=-1)) * 180/np.pi
            sidechain_torsion_preds = max_sampling(torch.stack(preds[3:9], axis=2),
                                    self.bin_references["angles"])
            angle_preds = np.concatenate([backbone_angle_preds, sidechain_torsion_preds], axis=-1)

        if self.bond_length_bin:
            n_ca_blen_preds = max_sampling(preds[9], self.bin_references["n_ca_bond_length"])
            ca_c_blen_preds = max_sampling(preds[10], self.bin_references["ca_c_bond_length"])
            c_n_blen_preds = max_sampling(preds[11], self.bin_references["c_n_bond_length"])
            blens_preds = np.stack([n_ca_blen_preds,
                                    ca_c_blen_preds,
                                    c_n_blen_preds], axis=-1)
        else:
            blens_preds = tensor_to_numpy(torch.cat(preds[9:], axis=-1))
        return angle_preds, blens_preds

    def predict_and_evaulate(self, batch):
        batch_inputs = {"phi": batch.angs[:, :, 0].to(self.device[0]),
                        "psi": batch.angs[:, :, 1].to(self.device[0]),
                        "omega": batch.angs[:, :, 2].to(self.device[0]),
                        "res_type": torch.argmax(batch.seqs, axis=-1).to(self.device[0]),
                        "lengths": batch.lengths}
        preds = self.model(batch_inputs)
        loss = self.loss_fn(preds, batch)

        angle_preds, blens_preds = self.convert_prediction(preds)
        backbone_angle_rmse = rmse(angle_preds[:, :, :3], batch.angs[:, :, 3:6] * 180 / np.pi, periodic=True)
        sidechain_torsion_rmse = rmse(angle_preds[:, :, 3:], batch.angs[:, :, 6:] * 180 / np.pi, periodic=True)
        blens_rmse = rmse(blens_preds, batch.blens)
        return {"loss": loss,
                "angle_preds": angle_preds,
                "blens_preds": blens_preds,
                "angle_targets": tensor_to_numpy(batch.angs[:,:,3:]),
                "blens_targets": tensor_to_numpy(batch.blens),
                "backbone_angle_rmse": backbone_angle_rmse,
                "sidechain_torsion_rmse": sidechain_torsion_rmse,
                "blens_rmse": blens_rmse,
                "ids": batch.pids}

    def train(self,
              epochs,
              train_dataloader,
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
        if self.multi_gpu:
            self.model = nn.DataParallel(self.model, device_ids=self.device)
        self._optimizer_to_device()

        running_val_loss = []
        last_test_epoch = 0
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
            self.model.train()
            self.optimizer.zero_grad()

            if not self.verbose:
                train_dataloader = tqdm(train_dataloader)
            
            s = 1
            for batch in train_dataloader:
                self.optimizer.zero_grad()
                result = self.predict_and_evaulate(batch)
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


                if self.verbose:
                    print(datetime.now(),
                        "Train: Epoch %i/%i - step %i - loss: %.5f - running_loss: %.5f"
                        % (self.epoch, epochs, s, current_loss, running_loss / s))

                
                s += 1

            running_loss /= (s-1)

            rmse_bb_angles = np.mean(rmse_bb_angles[-100:])
            rmse_sc_tor = np.mean(rmse_sc_tor[-100:])
            rmse_bonds = np.mean(rmse_bonds[-100:])

            # plots
            self.plot_grad_flow()

            # validation
            val_error = float("inf")
            if val_dataloader is not None and \
                self.epoch % self.check_val == 0:
                if self.verbose:
                    print("Start validation")
                val_results = self.validate(val_dataloader)
                val_error = val_results["loss"]

            if self.multi_gpu:
                save_model = self.model.module
            else:
                save_model = self.model

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
            if self.best_val_loss > val_error:
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
                    test_results = self.validate(test_dataloader)
                    test_error = test_results['loss']
                    test_bb_angle_rmse = test_results['bb_angle_rmse']
                    test_sc_tor_rmse = test_results['sc_tor_rmse']
                    test_blens_rmse = test_results['blens_rmse']
                    torch.save(test_results, os.path.join(self.val_out_path, 'test_results.pkl'))
                    
                    # save scatter plot
                    bb_angle_plot = plot_scatter(test_results['angle_preds'][:,:3], 
                                              test_results['angle_targets'][:,:3] * 180 / np.pi,
                                              "predictions",
                                              "targets",
                                              "backbone angles")
                    self.logger.log_figure(bb_angle_plot, "backbone_angle_corr", self.epoch, "test")
                    sc_angle_plot = plot_scatter(test_results['angle_preds'][:,3:], 
                                              test_results['angle_targets'][:,3:] * 180 / np.pi,
                                              "predictions",
                                              "targets",
                                              "sidechain torsions")
                    self.logger.log_figure(sc_angle_plot, "sidechain_torsion_corr", self.epoch, "test")
                    blens_plot = plot_scatter(test_results['blens_preds'],
                                              test_results['blens_targets'],
                                              "predictions",
                                              "targets",
                                              "bond lengths")
                    self.logger.log_figure(blens_plot, "blens_corr", self.epoch, "test")
                    last_test_epoch = self.epoch
                    # np.save(os.path.join(self.val_out_path, 'test_Ei_epoch%i'%self.epoch), outputs['Ei'])

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

            # logging
            if self.epoch % self.check_log == 0:

                for i, param_group in enumerate(
                        self.optimizer.param_groups):
                        # self.scheduler.optimizer.param_groups):
                    old_lr = float(param_group["lr"])

                self.logger.log_result({
                    "epoch": self.epoch,
                    "lr": old_lr,
                    "time": time.time() - t0,
                    "loss": running_loss,
                    "val_loss": val_error,
                    "test_loss": test_error,
                    "tr_err_bb_angle(RMSE)": rmse_bb_angles,
                    'val_err_bb_angle(RMSE)': val_results['bb_angle_rmse'],
                    'test_err_bb_angle(RMSE)': test_bb_angle_rmse,
                    "tr_err_sc_tor(RMSE)": rmse_sc_tor,
                    'val_err_sc_tor(RMSE)': val_results['sc_tor_rmse'],
                    'test_err_sc_tor(RMSE)': test_sc_tor_rmse,
                    'tr_err_blens(RMSE)': rmse_bonds,
                    'val_err_blens(RMSE)': val_results['blens_rmse'],
                    'test_err_blens(RMSE)': test_blens_rmse
                })

    def log_statistics(self, data_collection):
        with open(os.path.join(self.output_path, "stats.txt"), "w") as f:
            f.write("Train data: %d\n" % data_collection.train[2])
            f.write("Val data: %d\n" % data_collection.val[2])
            f.write("Test data: %d\n" % data_collection.test[2])
            f.write("Normalizer: %s\n" % str(data_collection.normalizer))
            f.write("Test target hash: %s\n" % data_collection.hash)
