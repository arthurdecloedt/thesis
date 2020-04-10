import logging as lg

import torch
import torch.nn as nn
import torch.optim as optim
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

fileHandler = lg.FileHandler("train_test_val_batched4.log")
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

tests.resp_arr = tests.resp_arr * tests.scale
tests.create_valsplit(0.8)
tsampler = dataprocessing.MultiSplitSampler(tests)
vsampler = dataprocessing.MultiSplitSampler(tests, False)

writer = SummaryWriter()

# tsampler = dataprocessing.MultiSampler(trainset)
# vsampler = dataprocessing.MultiSampler(trainset)

trainloader = torch.utils.data.DataLoader(tests, batch_size=1, num_workers=1, sampler=tsampler)
valloader = torch.utils.data.DataLoader(tests, batch_size=1, num_workers=1, sampler=vsampler)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001)

nets = [
    embed_nets.Mixed_Net().double(),
    embed_nets.Pooling_Net().double(),
    embed_nets.Pre_net().double()
]

data, resp = next(iter(valloader))

optimizers = [optim.SGD(n.parameters(), lr=0.0001) for n in nets]
for n in nets:
    print(n.name)
    writer.add_graph(n, data)
# cont = embed_nets.Net_Container(net, trainloader, optimizer, criterion, True, valloader, writer)

m_cont = embed_nets.Multi_Net_Container(nets, trainloader, optimizers, criterion, True, valloader, writer)

m_cont.train(100)
