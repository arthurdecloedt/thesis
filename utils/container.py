import logging as lg
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import embed_nets


# This class defines a container wrapping around a Pytorch Network,
# A net_container bundles the dataset, dataloaders, data samplers and the net
# It provides a training loop for the network
# This class was written especially ad hoc and it does what i wanted it to do and nothing more,
# it was written very iteratively and therefore is a bit messy, especially the training loop
class Net_Container:
    val_loader: Optional[DataLoader]
    dataloader: DataLoader
    s_writer: Optional[SummaryWriter]

    def __init__(self, net, dataloader, optimizer, criterion, validation=False, val_loader=None, s_writer=None,
                 vix=False):

        # if we use plus network we need an applicable dataset
        assert not (isinstance(net, embed_nets.PoolingNetPlus) ^ dataloader.dataset.plus)
        self.plus = isinstance(net, embed_nets.PoolingNetPlus)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.net = net
        self.validation = validation
        assert (not validation) or val_loader is not None
        self.val_loader = val_loader
        self.tensorboard = True
        if s_writer is None:
            self.tensorboard = False
        # s_writer will be used to log training progress to tensorboard
        self.s_writer = s_writer

        # use vix for illustration purposes or for training (not implemented yet)
        self.vix = False
        if vix:
            if not self.dataloader.dataset.vix:
                lg.error("tried to enable vix for a dataset without it")
            else:
                self.vix = True

    # Training loop for the model,
    def train(self, epochs):
        net = self.net
        dataloader = self.dataloader
        optimizer = self.optimizer
        criterion = self.criterion
        zero_loss = 0.0
        vix_loss = 0.0

        train_size = self.dataloader.__getattribute__('sampler')
        # These arrays will serve for tracking the distribution of predictions vs truth
        resps = np.full(len(self.val_loader.__getattribute__('sampler')), -20.)
        truths = np.full(len(self.val_loader.__getattribute__('sampler')), -20.)
        for epoch in range(epochs):
            epoch_len = 0
            running_loss = 0.0
            # the model need to be set to train mode to enable autograd
            net.train()
            optimizer.zero_grad()
            # perform each training step
            # networks are in double precision mode for comparison reasons
            for i, data in enumerate(dataloader):
                inputs, resp = data[0].double(), data[1].double()
                if self.plus:
                    # the plus dataset provides us with an additional input tensor:
                    # the monomodal text embedding data
                    plus_i = data[3].double()
                    outputs = net(inputs, plus_i)
                else:
                    outputs = net(inputs)

                # calculate loss and perform a backwards pass,
                loss = criterion(outputs.squeeze(), resp.squeeze())
                running_loss += loss.item()
                loss /= 10
                loss.backward()
                # our variable sized input requires our batch size to be one
                # for better learning performance we accumulate gradient
                if (i + 1) % 10 == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                epoch_len += 1
            lg.info('e: %d | %s training_loss: %.10f', epoch + 1, epoch_len, (running_loss / epoch_len))
            if self.tensorboard:
                # we log our training los as a normal tensorboard scalar
                self.s_writer.add_scalar("Loss/Train", (running_loss / epoch_len), epoch + 1)
            # evaluation of model at this point
            if self.validation:
                optimizer.zero_grad()
                net.eval()
                val_loss = 0.0
                val_size = 0

                # with torch.no_grad():
                for i, data in enumerate(self.val_loader, 0):
                    if self.vix:
                        inputs, resp, vix = data[0].double(), data[1].double(), data[2].double()
                    else:
                        inputs, resp = data[0].double(), data[1].double()

                    # see training
                    if self.plus:
                        plus_i = data[3].double()
                        outputs = net(inputs, plus_i)
                    else:
                        outputs = net(inputs)

                    # for some epoch we want to analyse the distribution of results
                    if epoch < 15 or epoch % 10 == 0:
                        resps[i] = outputs.squeeze().item()
                    loss = criterion(outputs.squeeze(), resp.squeeze())
                    val_loss += loss.item()
                    val_size += 1
                    # these losses are cst so we only have to calculate them once
                    if epoch == 0:
                        loss_zero = criterion(torch.tensor(0), resp.squeeze())
                        zero_loss += loss_zero.item()
                        truths[i] = resp.squeeze().item()
                        if self.vix:
                            loss_vix = criterion(vix.squeeze(), resp.squeeze())
                            vix_loss += loss_vix.item()
            # lg.info('min: %s, max: %s',min_out,max_out)
            lg.info('e: %d | %s val_loss:      %.10f', epoch + 1, val_size, (val_loss / val_size))
            if self.tensorboard:

                self.s_writer.add_scalar("Loss/Val", (val_loss / val_size), epoch + 1)

                # here we log all values together, this will allow us to show them better on tensorboard
                w_dict = {
                    "train": (running_loss / epoch_len),
                    "val": (val_loss / val_size),
                    'zero': (zero_loss / val_size)
                }
                if self.vix:
                    w_dict['vix'] = (vix_loss / val_size)
                self.s_writer.add_scalars("Loss/combined", w_dict, epoch + 1)
                if epoch < 15 or epoch % 10 == 0:
                    self.s_writer.add_histogram("Val/predictions", resps, global_step=epoch + 1)
                    self.s_writer.add_histogram("Val/predictions_pos", resps[resps > 0], global_step=epoch + 1)

                    self.s_writer.add_histogram("Val/truth", truths, global_step=epoch + 1)
                    self.s_writer.add_histogram("Val/truth_pos", truths[truths > 0], global_step=epoch + 1)

                    self.s_writer.flush()

# This was code to train multiple networks at the same time, but it's not very efficient,
# Replaced by just running multiple instances with multiple shell scripts
# class Multi_Net_Container(Net_Container):
#
#     def __init__(self, nets, dataloader, optimizers, criterions, validation=False, val_loader=None, s_writer=None):
#         assert isinstance(nets, (tuple, list))
#         super().__init__(nets, dataloader, optimizers, criterions, validation, val_loader, s_writer)
#         self.n_nets = len(nets)
#         self.max_namesize = 5
#         for net in nets:
#             if len(net.name) > self.max_namesize:
#                 self.max_namesize = len(net.name)
#
#     def train(self, epochs, split_loss=True):
#         nets = self.net
#         dataloader = self.dataloader
#         optimizers = self.optimizer
#         criterion = self.criterion
#         median = torch.tensor(self.dataloader.dataset.baselines[1])
#         median_loss = 0.0
#         zero_loss = 0.0
#
#         for epoch in range(epochs):
#
#             epoch_len = []
#             running_loss = []
#             total_loss = []
#             loss = []
#             for i in range(self.n_nets):
#                 nets[i].train()
#                 epoch_len.append(0)
#                 running_loss.append(0.0)
#                 total_loss.append(0.0)
#                 optimizers[i].zero_grad()
#                 loss.append(None)
#             for i, data in enumerate(dataloader, 0):
#                 inputs, resp = data[0].double(), data[1].double()
#                 for j in range(self.n_nets):
#                     outputs = nets[j](inputs)
#                     loss[j] = criterion(outputs.squeeze(), resp.squeeze())
#                     running_loss[j] += loss[j].item()
#                     if (i + 1) % 10 == 0:
#                         loss[j] = total_loss[j] * 0.9 + loss[j] * 0.1
#                         loss[j].backward()
#                         optimizers[j].step()
#                         optimizers[j].zero_grad()
#                         total_loss[j] = 0.0
#                     else:
#                         total_loss[j] += loss[j]
#                     epoch_len[j] += 1
#             if total_loss[0] != 0.0:
#                 for j in range(self.n_nets):
#                     loss[j] = total_loss[j] * 0.9 + loss[j] * 0.1
#                     loss[j].backward()
#                     optimizers[j].step()
#                     optimizers[j].zero_grad()
#             if self.validation:
#                 val_loss = []
#                 val_size = []
#                 for i in range(self.n_nets):
#                     nets[i].eval()
#                     val_loss.append(0.0)
#                     val_size.append(0)
#
#                 with torch.no_grad():
#                     for i, data in enumerate(self.val_loader, 0):
#                         inputs, resp = data[0].double(), data[1].double()
#                         for j in range(self.n_nets):
#                             outputs = nets[j](inputs)
#                             # out=outputs.squeeze().item()
#                             # if out < min_out:
#                             #     min_out=out
#                             # elif out > max_out:
#                             #     max_out = out
#                             loss[j] = criterion(outputs.squeeze(), resp.squeeze())
#                             val_loss[j] += loss[j].item()
#                             val_size[j] += 1
#                             if epoch == 0 and j == 0:
#                                 loss_median = criterion(median, resp.squeeze())
#                                 loss_zero = criterion(torch.tensor(0), resp.squeeze())
#                                 median_loss += loss_median.item()
#                                 zero_loss += loss_zero.item()
#                 # lg.info('min: %s, max: %s',min_out,max_out)
#                 # lg.info('e: %d | %s training_loss: %.10f', epoch + 1, val_size, (val_loss[] / val_size))
#                 if self.tensorboard:
#                     # self.s_writer.add_scalar("Loss/Val", (val_loss / val_size), epoch + 1)
#                     writerdict = {"median": (median_loss / val_size[0]),
#                                   'zero': (zero_loss / val_size[0])
#                                   }
#                     for i in range(self.n_nets):
#                         writerdict[nets[i].name + ": train"] = (running_loss[i] / epoch_len[0])
#                         writerdict[nets[i].name + ": val"] = (val_loss[i] / val_size[0])
#                         lg.info('%s e: %d | %s training_loss: %.10f',
#                                 str.ljust(nets[i].name, self.max_namesize, ' ')
#                                 , epoch + 1, epoch_len[0], (running_loss[i] / epoch_len[0]))
#
#                         lg.info('%s e: %d | %s val_loss:      %.10f',
#                                 str.ljust(nets[i].name, self.max_namesize, ' ')
#                                 , epoch + 1, val_size[0], (val_loss[i] / val_size[0]))
#                     self.s_writer.add_scalars("Loss/combined", writerdict, epoch + 1)
