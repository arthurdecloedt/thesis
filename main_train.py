import logging as lg

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import Sampler
from torch.utils.tensorboard import SummaryWriter

import dataprocessing
import embed_nets

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

net = embed_nets.Mixed_Net().double()
net.train()
with open('resources/preferences.yaml') as f:
    prefs = yaml.load(f, Loader=yaml.FullLoader)
# trainset = dataprocessing.MultiSet(prefs)
# lg.info("saving dataset")
# trainset.save()
lg.info("loading dataset")

trainset = dataprocessing.Multi_Set_Binned(True, prefs=prefs)
# tests.inner.resp_arr = tests.inner.resp_arr * tests.inner.scale
# tsampler = dataprocessing.MultiSplitSampler(trainset)
# vsampler = dataprocessing.MultiSplitSampler(trainset,False)
tsampler = dataprocessing.MultiBinSampler(trainset)
vsampler = tsampler.get_val_sampler()

writer = SummaryWriter()

# tsampler = dataprocessing.MultiSampler(trainset)
# vsampler = dataprocessing.MultiSampler(trainset)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, num_workers=1, sampler=tsampler)
valloader = torch.utils.data.DataLoader(trainset, batch_size=1, num_workers=1, sampler=vsampler)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001)

data, resp = next(iter(trainloader))
while data.shape[2] < 10:
    data, resp = next(iter(trainloader))
writer.add_graph(net, data)
# optimizers = [optim.SGD(n.parameters(), lr=0.0001) for n in nets]
optimizer = optim.SGD(net.parameters(), lr=0.0001)

cont = embed_nets.Net_Container(net, trainloader, optimizer, criterion, True, valloader, s_writer=writer)

cont.train(200)
# m_cont = embed_nets.Multi_Net_Container(nets, trainloader, optimizers, criterion, True, valloader, writer)
