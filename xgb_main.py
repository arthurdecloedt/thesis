import logging as lg
import sys
import traceback
import warnings

import xgboost
import yaml
from torch.utils.tensorboard import SummaryWriter

import multiset
from container import XG_Container


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
trainset = multiset.MultiSet(prefs, True)

xgb = xgboost.XGBRegressor()

cont = XG_Container(trainset, xgb, 0.8, False)
writer = SummaryWriter()

cont.train()
cont.results_tb(writer)