import logging as lg

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import Sampler
from torch.utils.tensorboard import SummaryWriter

import dataprocessing
import embed_nets

logFormatter = lg.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = lg.getLogger()
rootLogger.setLevel(lg.INFO)

fileHandler = lg.FileHandler("train_test_val_batched.log")
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = lg.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

net = embed_nets.Pre_Net_Text().double()
net.train()
with open('resources/preferences.yaml') as f:
    prefs = yaml.load(f, Loader=yaml.FullLoader)
trainset = dataprocessing.MultiSet(prefs)
trainset.create_valsplit(0.5)
tsampler = dataprocessing.MultiSplitSampler(trainset)
vsampler = dataprocessing.MultiSplitSampler(trainset, False)

writer = SummaryWriter()

# tsampler = dataprocessing.MultiSampler(trainset)
# vsampler = dataprocessing.MultiSampler(trainset)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, num_workers=4, sampler=tsampler)
valloader = torch.utils.data.DataLoader(trainset, batch_size=1, num_workers=4, sampler=vsampler)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

try:
    writer.add_graph(net)
except Exception:
    pass
cont = embed_nets.Net_Container(net, trainloader, optimizer, criterion, True, valloader, writer)

cont.train(100)
