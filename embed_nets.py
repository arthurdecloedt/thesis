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


class CombinedAdaptivePool(nn.Module):

    def __init__(self, n):
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(n)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(n)
        self.name = 'Combined_Adaptive_Pool_layer'

    def forward(self, x):
        x = torch.cat((self.avg_pool(x).squeeze(), self.max_pool(x).squeeze()))
        return x


class PadPoolLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.name = "Pad_Pool_Layer"
        self.pool_a = nn.AvgPool1d(11, 1, ceil_mode=True)
        self.pool_m = nn.MaxPool1d(11, 1, ceil_mode=True)

    def forward(self, x):
        xp = F.pad(x, (5, 5), 'circular')
        x = torch.cat((x, self.pool_m(xp), self.pool_a(xp)), 1)
        return x


class PlusPoolLayer(torch.jit.ScriptModule):

    def __init__(self):
        super().__init__()
        self.name = 'Plus_Pool_Layer'

    @torch.jit.script_method
    def forward(self, x_t, x):
        output_size = x.size(2)
        x_m = F.adaptive_max_pool1d(x_t, output_size=output_size)
        x_a = F.adaptive_avg_pool1d(x_t, output_size=output_size)
        return torch.cat((x_m, x_a, x), 1)



class Pooling_Net(nn.Module):
    def __init__(self, name="pooling_net", n_pp=4, n_aggr=1, emb_sz=28):
        super().__init__()
        self.c1 = nn.Conv1d(3 * emb_sz, 20, 1)

        self.pad_pools = nn.ModuleList([PadPoolLayer() for f in range(n_pp)])

        self.c2 = nn.Conv1d(60, 15, 1)
        self.c3 = nn.Conv1d(45, 10, 1)
        self.c4 = nn.Conv1d(30, 10, 1)

        self.aggregate = CombinedAdaptivePool(n_aggr)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(2 * n_aggr * (10 + 28), 10)
        self.out = nn.Linear(10, 1)

        self.name = name

    def forward(self, x):
        x = self.embed(x)
        x = self.aggregate(x)
        x = self.linear(x)
        x = F.relu(x)
        x = self.out(x)
        return x

    def embed(self, x):
        xo = x.clone()
        x = self.pad_pools[0](x)
        x = F.relu(self.c1(x))
        x = self.pad_pools[1](x)
        x = F.relu(self.c2(x))
        x = self.pad_pools[2](x)
        x = F.relu(self.c3(x))
        x = self.pad_pools[3](x)
        x = F.relu(self.c4(x))
        x = torch.cat((x, xo), 1)
        return x


class PoolingNetPlus(Pooling_Net):

    def __init__(self, name="pooling_net_plus", n_aggr=1):
        super().__init__(name, emb_sz=32)
        self.ct1 = nn.Conv1d(4, 2, 1)
        self.ct2 = nn.Conv1d(4, 3, 1)
        self.pool_plus = PlusPoolLayer()

        self.linear = nn.Linear(2 * n_aggr * (10 + 28 + 4 + 3), 10)

    def forward(self, x, x_t):
        x_tp = F.relu(self.ct1(x_t))
        x = self.pool_plus(x_tp, x)
        x = self.embed(x)
        x_t = F.relu(self.ct2(x_t))
        x = self.aggregate(x)
        x_t = self.aggregate(x_t)
        x = torch.cat((x, x_t))
        x = F.relu(self.linear(x))
        return self.out(x)

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


