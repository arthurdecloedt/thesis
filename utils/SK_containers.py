import logging as lg
import pickle
from math import ceil

# noinspection PyUnresolvedReferences
import mpl_toolkits.mplot3d
import numpy as np
import sklearn as sk
import xgboost as xgb
from bayes_opt import BayesianOptimization
from matplotlib import pyplot as plt, cm
from torch.utils.tensorboard import SummaryWriter

from utils.multiset import MultiSet, ContigSet


class SK_container:
    def __init__(self, dataset, regressor, split=0.8, temporal=True) -> None:
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

        self.regressor = regressor
        self.bo_results = None

    def train(self):
        x, y = self.dataset.get_contig()
        np.random.shuffle(self.t_inds)
        np.random.shuffle(self.v_inds)

        xt = x[self.t_inds]
        yt = y[self.t_inds]
        self.regressor.fit(xt, yt)

    def register_bo(self, hyperparam_ranges, tuning_f):
        self.bo = BayesianOptimization(tuning_f, hyperparam_ranges)

    def cv_hopt_bayesian(self, hyperparam_ranges, s_writer=SummaryWriter(), resname="bayesopt_res",
                         iters=100):

        bayes_opt = self.bo
        lg.info("initializing Bayesian Optimization with 10 points")
        bayes_opt.maximize(n_iter=0, init_points=10)
        #
        # fig, fig2 = self.create_figures(hyperparam_ranges, bayes_opt)
        #
        # s_writer.add_figure("predicted_function_scatter", fig2, global_step=0)
        # s_writer.add_figure("predicted_function", fig, global_step=0)
        lg.info("bayesopt initialized")
        n = int(ceil(iters / 10))
        for a in range(n):
            bayes_opt.maximize(n_iter=9, init_points=1)
            # fig, fig2 = self.create_figures(hyperparam_ranges, bayes_opt)

            # s_writer.add_figure("predicted_function_scatter", fig2, global_step=a + 1)
            # s_writer.add_figure("predicted_function", fig, global_step=a + 1)
            s_writer.flush()
            self.bo_points = bayes_opt.res
            if self.bo_results != bayes_opt.max or a % 5 == 0:
                lg.info(bayes_opt.max)
            self.bo_results = bayes_opt.max
            with open(resname, 'wb') as file:
                pickle.dump(self.bo_points, file)
            lg.info("evaluated %s points", a * 10 + 10)
        self.bo_results = bayes_opt.max

    def reload_progress(self, resname="bayesopt_res"):
        with open(resname, 'rb') as file:
            points = pickle.load(file)
            x_obs = np.array([list(res["params"].values()) for res in points])
            y_obs = np.array([[res["target"]] for res in points])

            for x, y in zip(x_obs, y_obs):
                self.bo.register(x, y[0])
            lg.info("registered %s points", len(y))

            self.bo_results = self.bo.max
            self.bo_points = self.bo.res

    def create_figures(self, hyperparam_ranges, bo):
        x_obs = np.array([list(res["params"].values()) for res in bo.res])
        y_obs = np.array([[res["target"]] for res in bo.res])
        bo._gp.fit(x_obs, y_obs)
        n_params = len(hyperparam_ranges.keys())
        n_res = len(bo.res)
        keys = list(bo.res[0]['params'].keys())
        grid_res = 500
        spaces = []
        for a in range(n_params):
            h_range = hyperparam_ranges[keys[a]]
            spaces.append(np.linspace(h_range[0], h_range[1], num=grid_res))
        grids = np.meshgrid(*spaces)
        gr_shape = grids[0].shape
        grids_e = [np.expand_dims(g, -1) for g in grids]
        grid = np.concatenate(grids_e, -1)
        grid = grid.reshape((-1, 2))
        mu, sigma = bo._gp.predict(grid, return_std=True)
        mu_r = np.reshape(mu, gr_shape)
        sigma_r = np.reshape(sigma, gr_shape)
        fig = plt.figure(1)
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(grids[0], grids[1], mu_r, label='Prediction', cmap=cm.coolwarm)
        surf._facecolors2d = surf._facecolors3d
        surf._edgecolors2d = surf._edgecolors3d
        ax.set_xlabel(keys[0])
        ax.set_ylabel(keys[1])
        ax.set_zlabel('MSE')

        ax.legend()
        fig2 = plt.figure(2)
        ax2 = fig2.gca(projection='3d')
        ax2.scatter(x_obs[:, 0], x_obs[:, 1], y_obs, c=y_obs.squeeze(), label='Sample', cmap=cm.coolwarm)
        ax2.set_xlabel(keys[0])
        ax2.set_ylabel(keys[1])
        ax2.set_zlabel('MSE')
        ax2.legend()
        return fig, fig2


class lr_Container(SK_container):
    def __init__(self, dataset, regressor, split=0.8, temporal=True) -> None:
        super().__init__(dataset, regressor, split, temporal)

    def results_tb(self, writer=SummaryWriter()):
        x, y = self.dataset.get_contig()
        xt = x[self.t_inds]
        yt = y[self.t_inds]

        xv = x[self.v_inds]
        yv = y[self.v_inds]
        totalpredict = self.regressor.predict(x)
        trainpredict = self.regressor.predict(xt)
        valpredict = self.regressor.predict(xv)

        writer.add_histogram("total/pred", totalpredict)
        writer.add_histogram("val/pred", valpredict)
        writer.add_histogram("train/pred", trainpredict)
        xgb.plot_importance(self.regressor)
        fig = plt.gcf()
        # # fig2.add_subplot(xgb.plot_tree(self.xgb,num_trees=5))
        #
        # writer.add_figure("importance", fig)
        # plt.close(fig)
        # xgb.plot_tree(self.regressor, num_trees=5)
        # fig = plt.gcf()
        # fig.set_size_inches(150, 100)
        # fig.savefig('runs/tree.png')
        msettl = sk.metrics.mean_squared_error(totalpredict, y)
        msetr = sk.metrics.mean_squared_error(trainpredict, yt)
        msev = sk.metrics.mean_squared_error(valpredict, yv)
        lg.info("ttl   mse loss: %s", msettl)
        lg.info("val   mse loss: %s", msev)
        lg.info("train mse loss: %s", msetr)

    def tcv_eval(self,folds=10,f_skip=5,writer=SummaryWriter()):
        tss = sk.model_selection.TimeSeriesSplit(folds)
        x, y = self.dataset.get_contig()
        m_list = np.zeros((folds-f_skip,3))
        for i, (train_i, test_i) in enumerate(tss.split(x)):
            if i < f_skip:
                continue
            regressor = self.regressor
            regressor.fit(x[train_i], y[train_i])
            pred = regressor.predict(x[test_i])
            truth = y[test_i]
            mse = sk.metrics.mean_squared_error(truth, pred)
            mae = sk.metrics.mean_absolute_error(truth,pred)
            rmse = sk.metrics.mean_squared_error(truth, pred,squared=False)
            m_list[i-f_skip,:]= [mse,rmse,mae]
            lg.info("val   mse loss: %s", mse)
            lg.info("val  rmse loss: %s", rmse)
            lg.info("val   mae loss: %s", mae)

        return m_list

class XG_Container(SK_container):
    regressor: xgb.XGBRegressor
    dataset: [MultiSet, ContigSet]

    def __init__(self, dataset, regressor, split=0.8, temporal=True) -> None:
        super().__init__(dataset, regressor, split, temporal)

    def register_bo_cv(self, hyperparam_ranges, folds=5, h_writer=SummaryWriter()):
        x, y = self.dataset.get_contig()

        def bo_tune_xgb(max_depth, gamma):
            # cv_result = xgb.cv(params, dtrain, num_boost_round=int(n_estimators), nfold=folds)
            tss = sk.model_selection.TimeSeriesSplit(folds)
            acc = 0.0
            for train_i, test_i in tss.split(x):
                regressor = xgb.XGBRegressor(max_depth=int(max_depth), gamma=gamma, n_jobs=34)
                regressor.fit(x[train_i], y[train_i])
                pred = regressor.predict(x[test_i])
                truth = y[test_i]
                acc += sk.metrics.mean_squared_error(truth, pred)
            # Return the negative MSE
            t_mse = acc / folds
            h_writer.add_hparams({'max_depth': int(max_depth), 'gamma': gamma},
                                 {'hparam/time_split_test_rmse': t_mse})
            return -1. * t_mse

        super().register_bo(hyperparam_ranges, bo_tune_xgb)

    def register_bo_tcv(self, hyperparam_ranges, folds=5, h_writer=SummaryWriter()):
        x, y = self.dataset.get_contig()

        dtrain = xgb.DMatrix(data=x, label=y, nthread=2)

        def bo_tune_xgb(max_depth, gamma, lr, drop):
            prms = {
                'booster': 'dart',
                "max_depth": int(max_depth),
                "gamma": gamma,
                'learning_rate': lr,
                'rate_drop': drop,
                'nthread': 9,
                'eval_metric': 'rmse',
                'tree_method' : 'gpu_hist'
            }

            # cv_result = xgb.cv(prms, dtrain, 70, nfold=folds)
            tss = sk.model_selection.TimeSeriesSplit(folds * 2)

            cv_result = xgb.cv(prms, dtrain, num_boost_round=200, nfold=folds, folds=list(tss.split(x))[folds:],
                               early_stopping_rounds=10)

            # Return the negative MSE
            rmse = cv_result['test-rmse-mean'].iloc[-1]
            return -1. * rmse

        super().register_bo(hyperparam_ranges, bo_tune_xgb)

    def cv_hyper_opt_bayesian(self, hyperparam_ranges=None, folds=5, s_writer=SummaryWriter(), resname="xgb_bo_res.p",
                              iterations=0, h_writer=None, restart=False):
        if h_writer is None:
            h_writer = s_writer
        if hyperparam_ranges is None:
            hyperparam_ranges = {'max_depth': (3, 15),
                                 'gamma': (0.01, 1.5)
                                 }
        self.cv_hopt_bayesian(hyperparam_ranges, s_writer, resname, iterations)

    def reload_progress(self, resname="xgb_bo_res.p"):
        super().reload_progress(resname)

    # def save_figs(self,hyperparam_ranges,bo,writer):
    #     f1, f2 = self.create_figures(hyperparam_ranges,bo)
    #     writer.add_figure("predicted_function_scatter", f2, global_step=1)
    #     writer.add_figure("predicted_function", f, global_step=1)
    def tcv(self, folds=10, writer=SummaryWriter()):
        # cv_result = xgb.cv(params, dtrain, num_boost_round=int(n_estimators), nfold=folds)
        tss = sk.model_selection.TimeSeriesSplit(folds)
        x, y = self.dataset.get_contig()
        m_list = []
        for i, (train_i, test_i) in enumerate(tss.split(x)):
            regressor = self.regressor
            regressor.fit(x[train_i], y[train_i])
            pred = regressor.predict(x[test_i])
            truth = y[test_i]
            mse = sk.metrics.mean_squared_error(truth, pred)
            m_list.append(mse)
            dict = {'MSE_%s' % self.regressor.max_depth: mse}
            writer.add_scalars("MSE_TCV", dict, i)
        return m_list

    def results_tb(self, writer=SummaryWriter()):
        x, y = self.dataset.get_contig()
        xt = x[self.t_inds]
        yt = y[self.t_inds]

        xv = x[self.v_inds]
        yv = y[self.v_inds]
        totalpredict = self.regressor.predict(x)
        trainpredict = self.regressor.predict(xt)
        valpredict = self.regressor.predict(xv)

        writer.add_histogram("total/pred", totalpredict)
        writer.add_histogram("val/pred", valpredict)
        writer.add_histogram("train/pred", trainpredict)
        xgb.plot_importance(self.regressor)
        fig = plt.gcf()
        # fig2.add_subplot(xgb.plot_tree(self.xgb,num_trees=5))

        writer.add_figure("importance", fig)
        plt.close(fig)
        xgb.plot_tree(self.regressor, num_trees=5)
        fig = plt.gcf()
        fig.set_size_inches(150, 100)
        fig.savefig('runs/tree.png')
        msettl = sk.metrics.mean_squared_error(totalpredict, y)
        msetr = sk.metrics.mean_squared_error(trainpredict, yt)
        msev = sk.metrics.mean_squared_error(valpredict, yv)
        lg.info("ttl   mse loss: %s", msettl)
        lg.info("val   mse loss: %s", msev)
        lg.info("train mse loss: %s", msetr)
