import logging as lg

import yaml
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter

from utils import container, embed_nets, multiset_plus

# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#     log = file if hasattr(file, 'write') else sys.stderr
#     traceback.print_stack(file=log)
#     log.write(warnings.formatwarning(message, category, filename, lineno, line))
#
#
# warnings.showwarning = warn_with_traceback

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

trainset = multiset_plus.MultiSetPlus(prefs, contig_resp=True)
try:
    trainset.log_embedding(SummaryWriter("embedding_mvso"))
except Exception as error:
    lg.warning("embedding failed")
    lg.warning(error)

cont = container.TS_validation_net_container(trainset, embed_nets.AttNet, nn.MSELoss, optim.Adam)

cont.perform_ts_val(75, folds=10, f_skip=5)
