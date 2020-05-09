import logging as lg
import sys
import traceback
import warnings

import xgboost
import yaml
from torch.utils.tensorboard import SummaryWriter

from utils import multiset
from utils.container import XG_Container


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
trainset = multiset.ContigSet()
xgb = xgboost.XGBRegressor()

cont = XG_Container(trainset, xgb, 0.8, False)
writer = SummaryWriter()

cont.cv_hyper_opt_bayesian(s_writer=writer)

cont.xgb = xgboost.XGBRegressor(n_estimators=int(cont.bo_results['params']['n_estimators']),
                                max_depth=int(cont.bo_results['params']['max_depth']))
cont.train()
cont.results_tb(writer)
# cont.train()
# cont.results_tb(writer)
