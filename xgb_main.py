import logging as lg
import sys
import traceback
import warnings
from matplotlib import pyplot as plt, cm

# noinspection PyUnresolvedReferences
import mpl_toolkits.mplot3d
import sklearn.linear_model as skl
import sklearn as sk
import xgboost
import yaml
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils import multiset
from utils.SK_containers import XG_Container,lr_Container


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


warnings.showwarning = warn_with_traceback

logFormatter = lg.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = lg.getLogger()
rootLogger.setLevel(lg.INFO)

fileHandler = lg.FileHandler("train_binned.log")
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = lg.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

with open('resources/preferences.yaml') as f:
    prefs = yaml.load(f, Loader=yaml.FullLoader)

lg.info("loading dataset")
# trainset = multiset.MultiSet(prefs,contig_resp=True)
# # trainset.save()
trainset = multiset.ContigSet()
xgb = xgboost.XGBRegressor()
lr = skl.LinearRegression()

cont = XG_Container(trainset,xgb)
writer = SummaryWriter()
# h_writer = SummaryWriter("hparam_part/hparam_xgb")
# cont.train()
# cont.cv_hyper_opt_bayesian(s_writer=writer,iterations=200,h_writer=h_writer,resname="xgb_bo_2.p")
hyperparam_ranges = {'max_depth': (3, 8),
                     'gamma': (0.001, 1.5),
                     'lr': (0.0001, 0.3),
                     'drop': (0, .3)
                     }

# cont.results_tb()
# cont.register_bo_tcv(hyperparam_ranges)
# list = cont.tcv_eval(10,5,writer)
# print(list)
# print(np.mean(list,0))
#     cont.bo.set_bounds(
#         {
#             'max_depth': (3 , 6),
#             "drop": (0,0.1)
#         }
# # )
# cont.reload_progress("xgb_bo_10_final.p")
# cont.cv_hyper_opt_bayesian(s_writer=writer, iterations=500, h_writer=h_writer, resname="xgb_bo_10_redo3.p")
# cont.train()

# lg.info(cont.bo_results)
# writer.flush()
x, y = cont.dataset.get_contig()
#

x_v = x[-20000:]
y_v = y[-20000:]
x_t = x[:-20000]
y_t = y[:-20000]
dtrain = xgboost.DMatrix(data=x_t, label=y_t, nthread=2)
deval = xgboost.DMatrix(data=x_v, label=y_v, nthread=2)

prms = {
     'booster': 'dart',
    "max_depth": 3,
    "gamma": 0.6343837174116629,
    'learning_rate': 0.30840599405756164,
    'rate_drop': 0.098535117986459,
    'nthread': 34
}
# folds = 5
# # # cv_result = xgboost.cv(prms, dtrain, 70, nfold=folds)
# tss = sk.model_selection.TimeSeriesSplit(5 * 2)
#
# cv_result = xgboost.cv(prms, dtrain, num_boost_round=1000, nfold=folds, folds=list(tss.split(x)),
#                    early_stopping_rounds=20,verbose_eval=True)
# print(cv_result)
# # cont.train()
bst = xgboost.train(prms,dtrain,evals=[(deval,'')],num_boost_round=500,early_stopping_rounds=10)
xgboost.plot_importance(bst,importance_type='gain')
fig = plt.gcf()

fig.savefig('fsc_gain.png')

xgboost.plot_importance(bst,importance_type='cover')
fig = plt.gcf()
fig.savefig('fsc_cover.png')

xgboost.plot_importance(bst)
fig = plt.gcf()
fig.savefig('fsc_weight.png')
