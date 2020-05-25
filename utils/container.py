import logging as lg
import math
import pickle
from copy import deepcopy
from functools import partial
from typing import Optional

import numpy as np
import scipy.stats as scistat
import sklearn.metrics as skm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import embed_nets, data_utils, multiset_plus
from utils.multiset import MultiSet, Multi_Set_Binned


class Net_Container:
    val_loader: Optional[DataLoader]
    dataloader: DataLoader
    s_writer: Optional[SummaryWriter]

    def __init__(self, net, dataloader, optimizer, criterion, validation=False, val_loader=None, s_writer=None,
                 vix=False, plus=False, multiloss=False):

        self.plus = plus
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.net = net
        self.validation = validation
        self.multi_loss = False
        if multiloss:
            assert net.multi_loss
            self.multi_loss = True
        assert (not validation) or val_loader is not None
        self.val_loader = val_loader
        self.tensorboard = False
        if s_writer is None:
            self.tensorboard = False
        self.s_writer = s_writer
        self.vix = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.lb = isinstance(net,embed_nets.Pooling_Net_Max)
        self.dataloader.dataset.lookback = self.lb
        if self.lb:
            self.dataloader.dataset.n_l = self.net.n_regr
        if vix:
            if not self.dataloader.dataset.vix:
                lg.error("tried to enable vix for a dataset without it")
            else:
                self.vix = True

    def evaluate(self, metrics=None, suffix=""):
        net = self.net
        net.eval()
        vlen = len(self.val_loader.__getattribute__('sampler'))
        preds_v = np.zeros((vlen,))
        truths_v = np.zeros((vlen,))
        tlen = len(self.dataloader.__getattribute__('sampler'))

        preds_t = np.zeros(tlen)
        truths_t = np.zeros(tlen)

        for i, data in enumerate(self.dataloader):
            inputs, resp = data[0].double().to(device=self.device), data[1].double().to(device=self.device)
            if self.lb:
                input2= data[3].double().to(device=self.device)
                inputs = (inputs,input2)
            if self.plus:
                plus_i = data[3].double().to(device=self.device)
                output = net(inputs, plus_i)
            else:
                output = net(inputs)
            if self.multi_loss:
                output = output[0]
            preds_t[i] = output
            truths_t[i] = resp
        for i, data in enumerate(self.val_loader):
            inputs, resp = data[0].double().to(device=self.device), data[1].double().to(device=self.device)
            if self.lb:
                input2= data[3].double().to(device=self.device)
                inputs = (inputs,input2)

            if self.plus:
                plus_i = data[3].double().to(device=self.device)
                output = net(inputs, plus_i)
            else:
                output = net(inputs)
            if self.multi_loss:
                output = output[0]
            preds_v[i] = output
            truths_v[i] = resp
        ds = self.dataloader.dataset
        ds: MultiSet
        preds_v = ds.unscale(preds_v)
        preds_t = ds.unscale(preds_t)
        truths_v = ds.unscale(truths_v)
        truths_t = ds.unscale(truths_t)

        def corp(x, y):
            return scistat.pearsonr(x, y)[0]

        def corsp(x, y):
            return scistat.spearmanr(x, y)[0]

        if metrics is None:
            metrics = {
                "MSE": skm.mean_squared_error,
                "MAE": skm.mean_absolute_error,
                "RMSE": partial(skm.mean_squared_error, squared=False),
                "Pearson Corr": corp,
                "Spearman Rank Corr": corsp
            }
        ev_res = {}
        tot_pred = np.concatenate((preds_t, preds_v)).flatten()
        tot_truth = np.concatenate((truths_t, truths_v)).flatten()

        # self.s_writer.add_histogram("train_multi_p", preds_t)
        # self.s_writer.add_histogram("train_multi_t", truths_t)
        # self.s_writer.add_histogram("ttl_multi_p", tot_pred)
        # self.s_writer.add_histogram("ttl_multi_t", tot_truth)
        # self.s_writer.add_histogram("val_multi_p", preds_v)
        # self.s_writer.add_histogram("val_multi_t", truths_v)
        autoregr = {}

        for name, fun in metrics.items():
            ev_res[name + '_train' + suffix] = fun(truths_t, preds_t)
            ev_res[name + '_val' + suffix] = fun(truths_v, preds_v)
            ev_res[name + '_ttl' + suffix] = fun(tot_truth, tot_pred)
            ev_res[name + '_train_r_self' + suffix] = fun(truths_t[1:], truths_t[:-1])
            ev_res[name + '_val_r_self' + suffix] = fun(truths_v[1:], truths_v[:-1])
            ev_res[name + '_ttl_r_self' + suffix] = fun(tot_truth[1:], tot_truth[:-1])

        return ev_res

    def train(self, epochs, load_best=True, supress_lg_info=False):
        print(self.dataloader.dataset.lookback)

        net = self.net
        median = torch.tensor(self.dataloader.dataset.baselines[1])
        median_loss = 0.0
        zero_loss = 0.0
        vix_loss = 0.0
        lglevel = lg.INFO
        if supress_lg_info:
            lglevel = lg.getLogger().level
            lg.getLogger().setLevel(lg.WARNING)
        min_val_loss = math.inf
        # vlen = len(self.val_loader.__getattribute__('sampler'))
        min_los_statedict = {}
        for epoch in range(epochs):
            epoch_len = 0
            running_loss = 0.0
            net.train()
            self.optimizer.zero_grad()
            for i, data in enumerate(self.dataloader, 0):
                inputs, resp = data[0].double().to(device=self.device), data[1].double().to(device=self.device)
                if self.lb:
                    input2 = data[3].double().to(device=self.device)
                    outputs = net((inputs, input2))

                elif self.plus:
                    plus_i = data[3].double().to(device=self.device)
                    outputs = net(inputs, plus_i)
                else:
                    outputs = net(inputs)
                if self.multi_loss:
                    loss_interm = self.criterion(outputs[1], resp.squeeze().expand_as(outputs[1]))
                    loss_final = self.criterion(outputs[0].squeeze(), resp.squeeze())
                    loss = loss_interm + loss_final
                else:
                    loss = self.criterion(outputs.squeeze(), resp.squeeze())
                running_loss += loss.item()
                loss /= 32
                loss.backward()
                if (i + 1) % 32 == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                epoch_len += 1
            lg.info('e: %d | %s training_loss: %.10f', epoch + 1, epoch_len, (running_loss / epoch_len))
            if self.tensorboard:
                self.s_writer.add_scalar("Loss/Train", (running_loss / epoch_len), epoch + 1)
            if self.validation:
                self.optimizer.zero_grad()
                net.eval()
                val_loss = 0.0
                val_size = 0

                # with torch.no_grad():
                for i, data in enumerate(self.val_loader, 0):
                    if self.vix:
                        inputs, resp, vix = data[0].double().to(device=self.device), data[1].double().to(device=self.device), data[2].double()
                    else:
                        inputs, resp = data[0].double().to(device=self.device), data[1].double().to(device=self.device)
                    if self.lb:
                        input2 = data[3].double().to(device=self.device)
                        inputs = (inputs, input2)

                    if self.plus:
                        plus_i = data[3].double().to(device=self.device)
                        outputs = net(inputs, plus_i)
                        if self.multi_loss:
                            outputs = outputs[0]
                    else:
                        outputs = net(inputs)
                        if self.multi_loss:
                            outputs = outputs[0]

                    loss = self.criterion(outputs.squeeze(), resp.squeeze())
                    val_loss += loss.item()
                    val_size += 1
                    if epoch == 0:
                        loss_median = self.criterion(median, resp.squeeze())
                        loss_zero = self.criterion(torch.tensor(0), resp.squeeze())
                        median_loss += loss_median.item()
                        zero_loss += loss_zero.item()
                        if self.vix:
                            loss_vix = self.criterion(vix.squeeze(), resp.squeeze())
                            vix_loss += loss_vix.item()
                # lg.info('min: %s, max: %s',min_out,max_out)
                lg.info('e: %d | %s val_loss:      %.10f', epoch + 1, val_size, (val_loss / val_size))
                if min_val_loss > val_loss:
                    net: nn.Module
                    min_los_statedict = deepcopy(net.state_dict())

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
        if supress_lg_info:
            lg.getLogger().setLevel(lglevel)

        if self.validation and load_best:
            net.load_state_dict(min_los_statedict)


class TS_validation_net_container():

    def __init__(self, dataset: MultiSet, net_c: nn.Module, crit_c, optimizer_c, net_arg=(), crit_arg=(),
                 opt_arg=(), binned=False, dataset_b=None):
        super().__init__()

        self.opt_arg = opt_arg
        self.dataset = dataset
        self.optimizer_c = optimizer_c
        self.net_arg = net_arg
        self.crit_arg = crit_arg
        self.net_c = net_c
        self.crit_c = crit_c
        self.results = []
        self.binned = binned
        if binned:
            if dataset_b is None:
                assert dataset.has_contig
                self.dataset_b = Multi_Set_Binned(dataset)
                lg.info("running in binned training mode")
                lg.info("Will use TSS hybrid binned generator")
            else:
                self.dataset_b = dataset_b

    def perform_ts_val(self, max_epochs, early_stopping=False, folds=10, f_skip=5, resultf=None, log_eval=True,
                       repeat=1):
        if early_stopping:
            lg.warning("early stopping not yet implemented")
        self.early_stopping = False
        sampler_gen = data_utils.MultiTSSampler_binned_hybrid_gen(self.dataset, self.dataset_b, folds, f_skip) \
            if self.binned else data_utils.MultiTSSampler_gen(self.dataset, folds, f_skip)

        assert folds > f_skip
        lg.info("starting walk forward validation for %s", self.net_c.__name__)
        lg.info("%s folds, %s first folds skipped -> %s actual runs", folds, f_skip, folds - f_skip)
        if resultf is None:
            resultf = "results_tsv_%s.p" % self.net_c.__name__
        else:
            resultf = resultf + '_%s.p' % self.net_c.__name__

        for ind, (train_sampler, val_sampler) in enumerate(sampler_gen.get_samplers(), 1):
            lg.info('initializing run %s', ind)
            trainloader = torch.utils.data.DataLoader(self.dataset_b if self.binned else self.dataset, batch_size=1,
                                                      num_workers=1, sampler=train_sampler)
            valloader = torch.utils.data.DataLoader(self.dataset, batch_size=1, num_workers=1, sampler=val_sampler)

            crit = self.crit_c(*self.crit_arg)
            temp_res = []
            mse_min = math.inf
            for r in range(repeat):

                net = self.net_c(*self.net_arg).double()
                net: torch.nn.Module
                opt = self.optimizer_c(net.parameters(), *self.opt_arg)
                s_writer = SummaryWriter()
                cont = Net_Container(net, trainloader, opt, crit, True, valloader, vix=False,
                                     plus=isinstance(net, embed_nets.PoolingNetPlus),
                                     multiloss=hasattr(net, 'multi_loss') and net.multi_loss,
                                     s_writer=s_writer)
                lg.info("starting run %s", ind)
                cont.train(epochs=max_epochs, supress_lg_info=True)
                lg.info("run % finished, results:", ind)
                ev_res = cont.evaluate(suffix="_tsv")
                # if log_eval:
                #     for name, val in ev_res.items():
                #         lg.info("%s : %s", name, val)
                temp_res.append(ev_res)
                # with open(resultf, 'wb') as file:
                #     pickle.dump(self.results, file)
            mse_min = math.inf
            min_res = -1
            for i, res in enumerate(temp_res):
                if res['MSE_val_tsv'] < mse_min:
                    mse_min = res['MSE_val_tsv']
                    min_res = i
            if min_res == -1:
                raise RuntimeWarning("could not get a min")

            self.results.append(temp_res[min_res])
            with open(resultf, 'wb') as file:
                pickle.dump(self.results, file)

        sums = dict.fromkeys(self.results[0].keys(), 0)
        for i, res in enumerate(self.results, 1):
            if log_eval:
                lg.info("iter %s results:", i)
            for name, val in res.items():
                if log_eval:
                    lg.info("%s : %s", name, val)
                sums[name] += val
        means = {}
        for name, val in sums.items():
            mval = val / len(self.results)
            means[name + ' _ mean'] = mval
            if log_eval:
                lg.info("mean %s : %s", name, mval)
        self.results.append(means)
        with open(resultf, 'wb') as file:
            pickle.dump(self.results, file)

        lg.info("tsv done")


class TS_validation_net_hyper():
    def __init__(self, prefs: dict):
        super().__init__()
        lg.info("configuring hyper container")
        lg.debug(prefs)
        crits = {
            'mse': torch.nn.MSELoss,
            'huber': torch.nn.SmoothL1Loss
        }
        opts = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
            'adamw': torch.optim.AdamW
        }
        self.prefs = prefs
        assert 'nets' in prefs
        if 'crit' in prefs:
            self.crit_c = crits[prefs["crit"][0]]
        else:
            self.crit_c = torch.nn.MSELoss
        if 'opt' in prefs:
            opt_args_pos = prefs['opt'][1] if prefs['opt'][1] is not None else ()
            opt_args_named = prefs['opt'][2] if prefs['opt'][2] is not None else {}
            assert isinstance(opt_args_pos, tuple) and isinstance(opt_args_named, dict)

            self.opt_c = partial(opts[prefs['opt'][0]],*opt_args_pos, **opt_args_named)
        else:
            self.opt_c = partial(torch.optim.SGD,0.001)
        self.net_l = []
        for net in prefs['nets']:
            net_c = getattr(embed_nets, net[0])
            net_args_pos = net[1] if net[1] is not None else ()
            net_args_named = net[2] if net[2] is not None else {}
            assert isinstance(net_args_pos, tuple) and isinstance(net_args_named, dict)
            net_c = partial(net_c, *net_args_pos, **net_args_named)
            net_c.__name__ = net_c.func.__name__
            self.net_l.append(net_c)

        lg.info('configuration done, got %s net_configurations', len(self.net_l))
        self.dataset_c = multiset_plus.MultiSetCombined(prefs, contig_resp=True)
        self.dataset_m = multiset_plus.MultiWrapper.construct(self.dataset_c)

    def perform_run(self, folds, f_skip, max_epochs, repeats=5):

        lg.info("performing hyper run")
        l = len(self.net_l)
        folds_t = folds * l * repeats
        epochs_t = folds_t * max_epochs
        lg.info("will perform %s validations, %s repeats %s folds, %s epochs", l, repeats, folds_t, epochs_t)
        for net in self.net_l:
            if hasattr(net.func, 'plus') and net.func.plus:
                set = self.dataset_c
            else:
                set = self.dataset_m
            tss_cont = TS_validation_net_container(set, net, self.crit_c, self.opt_c)

            tss_cont.perform_ts_val(max_epochs, folds=folds, f_skip=f_skip, repeat=repeats,
                                    resultf=self.prefs['resultf'] if 'resultf' in self.prefs else None)

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
