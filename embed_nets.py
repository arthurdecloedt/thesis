import torch
import torch.nn as nn
import torch.nn.functional as F


class Mixed_Net(nn.Module):
    def __init__(self, name="mixed_net"):
        super(Mixed_Net, self).__init__()

        self.c1 = nn.Conv1d(28, 10, 1)

        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(76, 10)
        self.out = nn.Linear(10, 1)
        self.name = name

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
        # y = torch.tensor(27.9350)
        # y.requires_grad = True
        # x = x*0 +y
        return x

    # def num_flat_features(self, x):
    #     size = x.size()[1:]  # all dimensions except the batch dimension
    #     num_features = 1
    #     for s in size:
    #         num_features *= s
    #     return num_features


class Pre_net(nn.Module):

    def __init__(self, name="pre_net"):
        super().__init__()
        self.c1 = nn.Conv1d(28, 28, 1)
        self.c2 = nn.Conv1d(28, 28, 1)
        self.c3 = nn.Conv1d(28, 28, 1)
        self.c4 = nn.Conv1d(28, 1, 1)
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.name = name

    def forward(self, x):
        xc = self.c1(x)
        xc = F.relu(xc)
        x = torch.add(x, xc)
        xc = self.c2(x)
        xc = F.relu(xc)

        x = torch.add(x, xc)
        xc = self.c3(x)
        xc = F.relu(xc)
        x = torch.add(x, xc)
        x = self.c4(x)
        x = self.avg_pool(x)
        return x


class Pooling_Net(nn.Module):
    def __init__(self, name="pooling_net"):
        super().__init__()
        self.c1 = nn.Conv1d(3 * 28, 20, 1)

        self.pool_a = nn.AvgPool1d(11, 1, 0, ceil_mode=True)
        self.pool_m = nn.MaxPool1d(11, 1, ceil_mode=True)

        self.c2 = nn.Conv1d(60, 15, 1)
        self.c3 = nn.Conv1d(45, 10, 1)
        self.c4 = nn.Conv1d(30, 10, 1)
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(76, 10)
        self.out = nn.Linear(10, 1)

        self.name = name

    def forward(self, x):
        xo = x
        xp = F.pad(x, (5, 5), 'circular')
        x = torch.cat((x, self.pool_m(xp), self.pool_a(xp)), 1)
        x = F.relu(self.c1(x))

        xp = F.pad(x, (5, 5), 'circular')
        x = torch.cat((x, self.pool_m(xp), self.pool_a(xp)), 1)
        x = F.relu(self.c2(x))

        xp = F.pad(x, (5, 5), 'circular')
        x = torch.cat((x, self.pool_m(xp), self.pool_a(xp)), 1)
        x = F.relu(self.c3(x))

        xp = F.pad(x, (5, 5), 'circular')
        x = torch.cat((x, self.pool_m(xp), self.pool_a(xp)), 1)
        x = F.relu(self.c4(x))

        x = torch.cat((x, xo), 1)
        x = torch.cat((self.avg_pool(x).squeeze(), self.max_pool(x).squeeze()))
        x = self.linear(x)
        x = F.relu(x)
        x = self.out(x)
        return x


class Pre_Net_Text(nn.Module):

    def __init__(self, name="pre_net_text"):
        super(Pre_Net_Text, self).__init__()

        super().__init__()
        self.c1 = nn.Conv1d(4, 10, 1)
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(28, 10)
        self.out = nn.Linear(10, 1)
        self.name = name

    def forward(self, x):
        x = x[:, :4]
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


