import logging as lg
from typing import Optional

import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
import mpl_toolkits.mplot3d
import numpy as np
import sklearn as sk
import torch
import xgboost as xgb
from bayes_opt import BayesianOptimization
from matplotlib import cm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import embed_nets
from multiset import MultiSet, ContigSet


class XG_Container:
    xgb: xgb.XGBRegressor
    dataset: [MultiSet ,ContigSet]


    def __init__(self, dataset, xgb, split=0.8, temporal=False) -> None:
        self.dataset = dataset
        assert split <= 1
        self.split = split
        assert self.dataset.has_contig
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
        self.bo_results = None
    def train(self):
        x, y = self.dataset.get_contig()
        np.random.shuffle(self.t_inds)
        np.random.shuffle(self.v_inds)

        xt = x[self.t_inds]
        yt = y[self.t_inds]

        xv = x[self.v_inds]
        yv = y[self.v_inds]

        self.xgb.fit(xt, yt, eval_set=[(xv, yv)], verbose=True)

    def cv_hyper_opt_bayesian(self, hyperparam_ranges=None, folds=5, s_writer=SummaryWriter()):

        if hyperparam_ranges is None:
            hyperparam_ranges = {'max_depth': (3, 20),
                                 'n_estimators': (10, 500)
                                 }
        x, y = self.dataset.get_contig()

        dtrain = xgb.DMatrix(data=x, label=y, nthread=2)

        def bo_tune_xgb(max_depth, n_estimators):
            params = {'max_depth': int(max_depth),
                      'n_estimators': int(n_estimators),
                      'subsample': 0.8,
                      'eta': 0.1,
                      'eval_metric': 'rmse',
                      'nthread': 34
                      }

            cv_result = xgb.cv(params, dtrain, num_boost_round=int(n_estimators), nfold=folds)
            # Return the negative RMSE
            return -1.0 * cv_result['test-rmse-mean'].iloc[-1]

        xgb_bo = BayesianOptimization(bo_tune_xgb, hyperparam_ranges)
        xgb_bo.probe([7, 20])
        xgb_bo.probe([7, 22])
        self.bo = xgb_bo

        xgb_bo.maximize(n_iter=0, init_points=0, acq='ei')
        self.bo_results = xgb_bo.max

        xgb_bo.maximize(n_iter=0, init_points=10, acq='ei')

        fig, fig2 = self.create_figures(hyperparam_ranges, xgb_bo)

        s_writer.add_figure("predicted_function_scatter", fig2, global_step=0)
        s_writer.add_figure("predicted_function", fig, global_step=0)

        for a in range(100):
            xgb_bo.maximize(n_iter=10, acq='ei')
            fig, fig2 = self.create_figures(hyperparam_ranges, xgb_bo)

            s_writer.add_figure("predicted_function_scatter", fig2, global_step=a + 1)
            s_writer.add_figure("predicted_function", fig, global_step=a + 1)
            s_writer.flush()

        self.bo_results = xgb_bo.max

    def create_figures(self, hyperparam_ranges, xgb_bo):
        x_obs = np.array([list(res["params"].values()) for res in xgb_bo.res])
        y_obs = np.array([[res["target"]] for res in xgb_bo.res])
        xgb_bo._gp.fit(x_obs, y_obs)
        n_params = len(hyperparam_ranges.keys())
        n_res = len(xgb_bo.res)
        keys = list(xgb_bo.res[0]['params'].keys())
        grid_res = 100
        spaces = []
        for a in range(n_params):
            h_range = hyperparam_ranges[keys[a]]
            spaces.append(np.linspace(h_range[0], h_range[1], num=grid_res))
        grids = np.meshgrid(*spaces)
        gr_shape = grids[0].shape
        grids_e = [np.expand_dims(g, -1) for g in grids]
        grid = np.concatenate(grids_e, -1)
        grid = grid.reshape((-1, 2))
        mu, sigma = xgb_bo._gp.predict(grid, return_std=True)
        mu_r = np.reshape(mu, gr_shape)
        sigma_r = np.reshape(sigma, gr_shape)
        fig = plt.figure(1)
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(grids[0], grids[1], mu_r, label='Prediction', cmap=cm.coolwarm)
        surf._facecolors2d = surf._facecolors3d
        surf._edgecolors2d = surf._edgecolors3d
        ax.set_xlabel(keys[0])
        ax.set_ylabel(keys[1])
        ax.set_zlabel('-RSME')

        ax.legend()
        fig2 = plt.figure(2)
        ax2 = fig2.gca(projection='3d')
        print(y_obs)
        print(y_obs.shape)
        ax2.scatter(x_obs[:, 0], x_obs[:, 1], y_obs, c=y_obs.squeeze(), label='Sample', cmap=cm.coolwarm)
        ax2.set_xlabel(keys[0])
        ax2.set_ylabel(keys[1])
        ax2.set_zlabel('-RSME')
        ax2.legend()

        return fig, fig2

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
        xgb.plot_importance(self.xgb)
        fig = plt.gcf()
        # fig2.add_subplot(xgb.plot_tree(self.xgb,num_trees=5))

        writer.add_figure("importance", fig)
        plt.close(fig)
        xgb.plot_tree(self.xgb, num_trees=5)
        fig = plt.gcf()
        fig.set_size_inches(150, 100)
        fig.savefig('runs/tree.png')
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
                 vix=False, plus=False):

        assert isinstance(net, embed_nets.PoolingNetPlus) ^ (not plus)
        assert dataloader.dataset.plus ^ (not plus)
        self.plus = plus
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
                if self.plus:
                    plus_i = data[3].double()
                    outputs = net(inputs, plus_i)
                else:
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
                    if self.plus:
                        plus_i = data[3].double()
                        outputs = net(inputs, plus_i)
                    else:
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
