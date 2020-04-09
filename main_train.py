import logging as lg
import sys
import traceback
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Sampler
from torch.utils.tensorboard import SummaryWriter

import dataprocessing
import embed_nets


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


warnings.showwarning = warn_with_traceback

logFormatter = lg.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = lg.getLogger()
rootLogger.setLevel(lg.INFO)

fileHandler = lg.FileHandler("train_test_val_batched3.log")
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = lg.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

net = embed_nets.Mixed_Net().double()
net.train()
# with open('resources/preferences.yaml') as f:
#     prefs = yaml.load(f, Loader=yaml.FullLoader)
# trainset = dataprocessing.MultiSet(prefs)
# lg.info("saving dataset")
# trainset.save()
lg.info("loading dataset")
tests = dataprocessing.MultiSet.from_file()

tests.create_temporal_valsplit(0.75)
tsampler = dataprocessing.MultiSplitSampler(tests)
vsampler = dataprocessing.MultiSplitSampler(tests, False)

writer = SummaryWriter()

# tsampler = dataprocessing.MultiSampler(trainset)
# vsampler = dataprocessing.MultiSampler(trainset)

trainloader = torch.utils.data.DataLoader(tests, batch_size=1, num_workers=1, sampler=tsampler)
valloader = torch.utils.data.DataLoader(tests, batch_size=1, num_workers=1, sampler=vsampler)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001)

try:
    writer.add_graph(net)
except Exception:
    pass
cont = embed_nets.Net_Container(net, trainloader, optimizer, criterion, True, valloader, writer)

cont.train(100)
