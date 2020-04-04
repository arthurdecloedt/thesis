import torch
import torch.nn as nn
import torch.nn.functional as F


class Pre_Net(nn.Module):

    def __init__(self):
        super(Pre_Net, self).__init__()

        self.c1 = nn.Conv1d(28, 10, 1)
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)

        self.linear = nn.Linear(76, 10)
        self.out = nn.Linear(10, 1)

    def forward(self, x):
        xc = self.c1(x)

        x = torch.cat((xc, x), 1)
        x = F.relu(x)
        x1 = self.avg_pool(x)
        x2 = self.max_pool(x)
        x = torch.cat((x1.squeeze(), x2.squeeze()))
        x = self.linear(x)
        x = F.relu(x)
        x = self.out(x)

        return x

    # def num_flat_features(self, x):
    #     size = x.size()[1:]  # all dimensions except the batch dimension
    #     num_features = 1
    #     for s in size:
    #         num_features *= s
    #     return num_features



class Net_Container:
    def __init__(self, net, dataloader, optimizer, criterion, ):
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.optimizer.zero_grad()
        self.net = net
