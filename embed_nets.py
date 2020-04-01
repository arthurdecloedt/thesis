import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Sampler, Dataset
from torchvision import transforms
import datetime



class Pre_Net(nn.Module):

    def __init__(self):
        super(Pre_Net, self).__init__()
        self.fc1 = nn.Linear(28, 28)
        self.fc2 = nn.Linear(28, 10)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



class Net_Container:
    def __init__(self,net, dataloader, optimizer, criterion, Gpu = False):
        if Gpu :
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                print("Using GPU")

            else:
                print("Cuda is not available on this machine")
                print("Defaulting to Cpu")
                device = torch.device("cuda:0")
        else:
            device = torch.device("cuda:0")
            print("Using CPU")
        self.device = device
        self.net = net.to(device)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.optimizer.zero_grad()

class Multimodal(Dataset):
    def __init__(self, root='train'):
        self.root = root
        self.paths = [f.path for f in os.scandir(root) if f.is_file()]
        names = [f.name for f in os.scandir(root) if f.is_file()]

        self.ids = [f.split('.')[0] for f in names]

        self.ids.sort()

    def __len__(self):
        # Here, we need to return the number of samples in this dataset.
        return len(self.paths)

    def __getitem__(self, index):

        return image, index

class IdSampler(Sampler):

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(self.data_source.ids)

    def __len__(self):
        return len(self.data_source)