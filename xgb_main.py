import logging as lg
import sys
import traceback
import warnings

# noinspection PyUnresolvedReferences
import mpl_toolkits.mplot3d
import sklearn.linear_model as skl
import xgboost
import yaml
from torch.utils.tensorboard import SummaryWriter

from utils import multiset
from utils.SK_containers import XG_Container


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
# trainset = multiset.MultiSet(prefs, True)
# trainset.save()
trainset = multiset.ContigSet()
xgb = xgboost.XGBRegressor()
lr = skl.LinearRegression()

cont = XG_Container(trainset, xgb, 0.8, True)
writer = SummaryWriter()
h_writer = SummaryWriter("hparam_part/hparam_xgb")
# cont.train()
# cont.cv_hyper_opt_bayesian(s_writer=writer,iterations=200,h_writer=h_writer,resname="xgb_bo_2.p")
hyperparam_ranges = {'max_depth': (3, 8),
                     'gamma': (0.001, 5),
                     'lr': (0.0001, 1),
                     'drop': (0, .3)
                     }

cont.register_bo_tcv(hyperparam_ranges)
#     cont.bo.set_bounds(
#         {
#             'max_depth': (3 , 6),
#             "drop": (0,0.1)
#         }
# # )
cont.reload_progress("xgb_bo_9_final.p")
cont.cv_hyper_opt_bayesian(s_writer=writer, iterations=500, resname="xgb_bo_10_final.p")

lg.info(cont.bo_results)
writer.flush()
# cont.train()
# f1,f2 = cont.create_figures(hyperparam_ranges,cont.bo)
# writer.add_figure("predicted_function_scatter", f2, global_step=1)
# writer.add_figure("predicted_function", f1, global_step=1)
# writer.flush()
# cont.results_tb(writer)
# cont.train()
# cont.results_tb(writer)
