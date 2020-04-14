import logging as lg
from typing import Optional

import numpy as np
import sklearn as sk
import torch
import xgboost as xgb
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from multiset import MultiSet


class XG_Container:
    xgb: xgb.XGBRegressor
    dataset: MultiSet

    def __init__(self, dataset, xgb, split=0.8, temporal=False) -> None:
        self.dataset = dataset
        assert split <= 1
        self.split = split
        assert self.dataset.cr
        self.total_size = np.sum(self.dataset.contig_usable)
        self.train_size = int(self.total_size * self.split)
        self.val_size = self.total_size - self.train_size
        if temporal:
            inds = np.arange(self.total_size)
        else:
            inds = np.random.permutation(self.total_size)
        self.t_inds = inds[self.train_size:]
        self.v_inds = inds[:self.train_size]

        self.xgb = xgb

    def train(self):
        x, y = self.dataset.get_contig()
        np.random.shuffle(self.t_inds)
        np.random.shuffle(self.v_inds)

        xt = x[self.t_inds]
        yt = y[self.t_inds]

        xv = x[self.v_inds]
        yv = y[self.v_inds]

        self.xgb.fit(xt, yt, eval_set=[(xv, yv)], verbose=True)

    def results_tb(self, writer=SummaryWriter()):
        x, y = self.dataset.get_contig()
        xt = x[self.t_inds]
        yt = y[self.t_inds]

        xv = x[self.v_inds]
        yv = y[self.v_inds]
        totalpredict = self.xgb.predict(x)
        trainpredict = self.xgb.predict(xt)
        valpredict = self.xgb.predict(xv)

        writer.add_histogram("total/pred", totalpredict)
        writer.add_histogram("val/pred", valpredict)
        writer.add_histogram("train/pred", trainpredict)

        msettl = sk.metrics.mean_squared_error(totalpredict, y)
        msetr = sk.metrics.mean_squared_error(trainpredict, yt)
        msev = sk.metrics.mean_squared_error(valpredict, yv)
        lg.info("ttl   mse loss: %s", msettl)
        lg.info("val   mse loss: %s", msev)
        lg.info("train mse loss: %s", msetr)


class Net_Container:
    val_loader: Optional[DataLoader]
    dataloader: DataLoader
    s_writer: Optional[SummaryWriter]

    def __init__(self, net, dataloader, optimizer, criterion, validation=False, val_loader=None, s_writer=None,
                 vix=False):
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
        self.s_writer = s_writer
        self.vix = False
        if vix:
            if not self.dataloader.dataset.vix:
                lg.error("tried to enable vix for a dataset without it")
            else:
                self.vix = True

    def train(self, epochs):
        net = self.net
        dataloader = self.dataloader
        optimizer = self.optimizer
        criterion = self.criterion
        median = torch.tensor(self.dataloader.dataset.baselines[1])
        median_loss = 0.0
        zero_loss = 0.0
        vix_loss = 0.0
        resps = np.full(len(self.val_loader.__getattribute__('sampler')), -20.)
        truths = np.full(len(self.val_loader.__getattribute__('sampler')), -20.)
        vlen = len(self.val_loader.__getattribute__('sampler'))
        for epoch in range(epochs):
            testloss = 0.0

            epoch_len = 0
            running_loss = 0.0
            net.train()
            optimizer.zero_grad()
            for i, data in enumerate(dataloader, 0):
                inputs, resp = data[0].double(), data[1].double()
                outputs = net(inputs)
                loss = criterion(outputs.squeeze(), resp.squeeze())
                running_loss += loss.item()
                loss /= 10
                loss.backward()
                if (i + 1) % 10 == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                epoch_len += 1
            lg.info('e: %d | %s training_loss: %.10f', epoch + 1, epoch_len, (running_loss / epoch_len))
            if self.tensorboard:
                self.s_writer.add_scalar("Loss/Train", (running_loss / epoch_len), epoch + 1)
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

                    outputs = net(inputs)
                    if epoch < 15 or epoch % 10 == 0:
                        resps[i] = outputs.squeeze().item()
                    # out=outputs.squeeze().item()
                    # if out < min_out:
                    #     min_out=out
                    # elif out > max_out:
                    #     max_out = out
                    loss = criterion(outputs.squeeze(), resp.squeeze())
                    val_loss += loss.item()
                    val_size += 1
                    if epoch == 0:
                        loss_median = criterion(median, resp.squeeze())
                        loss_zero = criterion(torch.tensor(0), resp.squeeze())
                        median_loss += loss_median.item()
                        zero_loss += loss_zero.item()
                        truths[i] = resp.squeeze().item()
                        if self.vix:
                            loss_vix = criterion(vix.squeeze(), resp.squeeze())
                            vix_loss += loss_vix.item()
            # lg.info('min: %s, max: %s',min_out,max_out)
            lg.info('e: %d | %s val_loss:      %.10f', epoch + 1, val_size, (val_loss / val_size))
            if self.tensorboard:
                self.s_writer.add_scalar("Loss/Val", (val_loss / val_size), epoch + 1)
                w_dict = {
                    "train": (running_loss / epoch_len),
                    "val": (val_loss / val_size),
                    "median": (median_loss / val_size),
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
                    if epoch == 0:
                        print(median_loss / val_size)
                        print(median)


class Multi_Net_Container(Net_Container):

    def __init__(self, nets, dataloader, optimizers, criterions, validation=False, val_loader=None, s_writer=None):
        assert isinstance(nets, (tuple, list))
        super().__init__(nets, dataloader, optimizers, criterions, validation, val_loader, s_writer)
        self.n_nets = len(nets)
        self.max_namesize = 5
        for net in nets:
            if len(net.name) > self.max_namesize:
                self.max_namesize = len(net.name)

    def train(self, epochs, split_loss=True):
        nets = self.net
        dataloader = self.dataloader
        optimizers = self.optimizer
        criterion = self.criterion
        median = torch.tensor(self.dataloader.dataset.baselines[1])
        median_loss = 0.0
        zero_loss = 0.0

        for epoch in range(epochs):

            epoch_len = []
            running_loss = []
            total_loss = []
            loss = []
            for i in range(self.n_nets):
                nets[i].train()
                epoch_len.append(0)
                running_loss.append(0.0)
                total_loss.append(0.0)
                optimizers[i].zero_grad()
                loss.append(None)
            for i, data in enumerate(dataloader, 0):
                inputs, resp = data[0].double(), data[1].double()
                for j in range(self.n_nets):
                    outputs = nets[j](inputs)
                    loss[j] = criterion(outputs.squeeze(), resp.squeeze())
                    running_loss[j] += loss[j].item()
                    if (i + 1) % 10 == 0:
                        loss[j] = total_loss[j] * 0.9 + loss[j] * 0.1
                        loss[j].backward()
                        optimizers[j].step()
                        optimizers[j].zero_grad()
                        total_loss[j] = 0.0
                    else:
                        total_loss[j] += loss[j]
                    epoch_len[j] += 1
            if total_loss[0] != 0.0:
                for j in range(self.n_nets):
                    loss[j] = total_loss[j] * 0.9 + loss[j] * 0.1
                    loss[j].backward()
                    optimizers[j].step()
                    optimizers[j].zero_grad()
            if self.validation:
                val_loss = []
                val_size = []
                for i in range(self.n_nets):
                    nets[i].eval()
                    val_loss.append(0.0)
                    val_size.append(0)

                with torch.no_grad():
                    for i, data in enumerate(self.val_loader, 0):
                        inputs, resp = data[0].double(), data[1].double()
                        for j in range(self.n_nets):
                            outputs = nets[j](inputs)
                            # out=outputs.squeeze().item()
                            # if out < min_out:
                            #     min_out=out
                            # elif out > max_out:
                            #     max_out = out
                            loss[j] = criterion(outputs.squeeze(), resp.squeeze())
                            val_loss[j] += loss[j].item()
                            val_size[j] += 1
                            if epoch == 0 and j == 0:
                                loss_median = criterion(median, resp.squeeze())
                                loss_zero = criterion(torch.tensor(0), resp.squeeze())
                                median_loss += loss_median.item()
                                zero_loss += loss_zero.item()
                # lg.info('min: %s, max: %s',min_out,max_out)
                # lg.info('e: %d | %s training_loss: %.10f', epoch + 1, val_size, (val_loss[] / val_size))
                if self.tensorboard:
                    # self.s_writer.add_scalar("Loss/Val", (val_loss / val_size), epoch + 1)

                    writerdict = {"median": (median_loss / val_size[0]),
                                  'zero': (zero_loss / val_size[0])
                                  }
                    for i in range(self.n_nets):
                        writerdict[nets[i].name + ": train"] = (running_loss[i] / epoch_len[0])
                        writerdict[nets[i].name + ": val"] = (val_loss[i] / val_size[0])
                        lg.info('%s e: %d | %s training_loss: %.10f',
                                str.ljust(nets[i].name, self.max_namesize, ' ')
                                , epoch + 1, epoch_len[0], (running_loss[i] / epoch_len[0]))

                        lg.info('%s e: %d | %s val_loss:      %.10f',
                                str.ljust(nets[i].name, self.max_namesize, ' ')
                                , epoch + 1, val_size[0], (val_loss[i] / val_size[0]))
                    self.s_writer.add_scalars("Loss/combined", writerdict, epoch + 1)
