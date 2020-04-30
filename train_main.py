import logging as lg

import torch
import yaml
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter

import container
import dataprocessing
import embed_nets
import multiset_plus

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

net = embed_nets.PoolingNetPlus().double()
# net = embed_nets.Pooling_Net().double()

net.train()
# net2.train()
with open('resources/preferences.yaml') as f:
    prefs = yaml.load(f, Loader=yaml.FullLoader)
    lg.info("loading dataset")

trainset = multiset_plus.MultiSetCombined(prefs, n_procs=4)
# lg.info("saving dataset")
# trainset.save()

# trainset = multiset.Multi_Set_Binned(True, prefs=prefs)
# tsampler = dataprocessing.MultiBinSampler(trainset)
# vsampler = tsampler.get_val_sampler(.8)

# trainset = multiset.MultiSet(prefs)
trainset.create_valsplit(.8)
tsampler = dataprocessing.MultiSplitSampler(trainset, True)
vsampler = dataprocessing.MultiSplitSampler(trainset, False)

writer = SummaryWriter()

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, num_workers=0, sampler=tsampler)
valloader = torch.utils.data.DataLoader(trainset, batch_size=1, num_workers=0, sampler=vsampler)
#
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001)

# data, resp, _ = next(iter(trainloader))
# while data.shape[2] < 10:
#     data, resp, _ = next(iter(trainloader))
# writer.add_graph(net, data)

data = (torch.ones((1, 28, 50)).double(), torch.ones((1, 4, 500)).double())
data1 = torch.ones((1, 28, 50)).double()

writer.add_graph(net, data)
writer.flush()
# optimizers = [optim.SGD(n.parameters(), lr=0.0001) for n in nets]

cont = container.Net_Container(net, trainloader, optimizer, criterion, True, valloader, s_writer=writer, vix=True,
                               plus=True)
# #

cont.train(200)
# # m_cont = embed_nets.Multi_Net_Container(nets, trainloader, optimizers, criterion, True, valloader, writer)
