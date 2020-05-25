import pickle
import numpy as np

import logging as lg

import torch
import yaml
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter

from utils import data_utils, container, embed_nets, multiset_plus

with open('resources/preferences.yaml') as f:
    prefs = yaml.load(f, Loader=yaml.FullLoader)
    lg.info("loading dataset")

trainset = multiset_plus.MultiSetPlus(prefs, contig_resp=True)

s_writer = SummaryWriter("embedding_b500")
trainset.log_embedding(s_writer)
