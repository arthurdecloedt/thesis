import logging as lg
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Mixed_Net(nn.Module):
    def __init__(self, name="mixed_net"):
        super(Mixed_Net, self).__init__()

        self.c1 = nn.Conv1d(28, 10, 1)
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)
        self.flatten = nn.res
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
        self.c1 = nn.Conv1d(56, 20, 1)

        self.pool1 = nn.AvgPool1d(11, 1, 0, ceil_mode=True)
        self.c2 = nn.Conv1d(40, 10, 1)
        self.c3 = nn.Conv1d(40, 10, 1)
        self.c4 = nn.Conv1d(20, 1, 1)
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.name = name

    def forward(self, x):
        if x.shape[2] > 5:
            xp = F.pad(x, (5, 5), 'circular')
        else:
            xp = F.pad(x, (5, 5), 'replicate')

        xp = self.pool1(xp)
        x = torch.cat((x, xp), 1)

        x = F.relu(self.c1(x))

        if x.shape[2] > 5:
            xp = F.pad(x, (5, 5), 'circular')
        else:
            xp = F.pad(x, (5, 5), 'replicate')

        xp = self.pool1(xp)
        x = torch.cat((x, xp), 1)
        x = F.relu(self.c2(x))

        if x.shape[2] > 5:
            xp = F.pad(x, (5, 5), 'circular')
        else:
            xp = F.pad(x, (5, 5), 'replicate')

        xp = self.pool1(xp)
        x = torch.cat((x, xp), 1)
        x = F.relu(self.c3(x))

        x = self.avg_pool(self.c4(x))
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


class Net_Container:
    dataloader: DataLoader
    s_writer: Optional[SummaryWriter]

    def __init__(self, net, dataloader, optimizer, criterion, validation=False, val_loader=None, s_writer=None):
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.net = net
        self.validation = validation
        assert (not validation) or val_loader is not None
        self.val_loader = val_loader
        self.tensorboard = True
        if s_writer is None:
            self.tensorboard = False
        self.s_writer = s_writer

    def train(self, epochs):
        net = self.net
        dataloader = self.dataloader
        optimizer = self.optimizer
        criterion = self.criterion
        median = torch.tensor(self.dataloader.dataset.baselines[1])
        median_loss = 0.0
        zero_loss = 0.0
        for epoch in range(epochs):

            epoch_len = 0
            running_loss = 0.0
            total_loss = 0.
            net.train()
            optimizer.zero_grad()
            loss = None
            for i, data in enumerate(dataloader, 0):
                inputs, resp = data[0].double(), data[1].double()
                outputs = net(inputs)
                loss = criterion(outputs.squeeze(), resp.squeeze())
                running_loss += loss.item()
                if (i + 1) % 20 == 0:
                    loss = total_loss * 0.9 + loss * 0.1
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss = 0.0
                else:
                    total_loss += loss
                epoch_len += 1
            if total_loss != 0.0:
                loss = total_loss * 0.9 + loss * 0.1
                loss.backward()
            lg.info('e: %d | %s training_loss: %.10f', epoch + 1, epoch_len, (running_loss / epoch_len))
            if self.tensorboard:
                self.s_writer.add_scalar("Loss/Train", (running_loss / epoch_len), epoch + 1)
            if self.validation:
                optimizer.zero_grad()
                # net.eval()
                val_loss = 0.0
                val_size = 0
                with torch.no_grad():
                    for i, data in enumerate(self.val_loader, 0):
                        inputs, resp = data[0].double(), data[1].double()
                        outputs = net(inputs)
                        # out=outputs.squeeze().item()
                        # if out < min_out:
                        #     min_out=out
                        # elif out > max_out:
                        #     max_out = out
                        loss = criterion(outputs.squeeze(), resp.squeeze())
                        val_loss += loss.item()
                        val_size += 1
                        if epoch == 0:
                            loss_median = criterion(median, resp.squeeze())
                            loss_zero = criterion(torch.tensor(0), resp.squeeze())
                            median_loss += loss_median.item()
                            zero_loss += loss_zero.item()
                # lg.info('min: %s, max: %s',min_out,max_out)
                lg.info('e: %d | %s val_loss:      %.10f', epoch + 1, val_size, (val_loss / val_size))
                if self.tensorboard:
                    self.s_writer.add_scalar("Loss/Val", (val_loss / val_size), epoch + 1)
                    self.s_writer.add_scalars("Loss/combined", {
                        "train": (running_loss / epoch_len),
                        "val": (val_loss / val_size),
                        "median": (median_loss / val_size),
                        'zero': (zero_loss / val_size)
                    }, epoch + 1)


class Multi_Net_Container(Net_Container):

    def __init__(self, nets, dataloader, optimizers, criterions, validation=False, val_loader=None, s_writer=None):
        assert isinstance(nets, (tuple, list))
        super().__init__(nets, dataloader, optimizers, criterions, validation, val_loader, s_writer)
        self.n_nets = len(nets)
        self.max_namesize = 5
        for net in nets:
            if len(net.name) > self.max_namesize:
                self.max_namesize = len(net.name)

    def train(self, epochs, split_loss=True):
        nets = self.net
        dataloader = self.dataloader
        optimizers = self.optimizer
        criterion = self.criterion
        median = torch.tensor(self.dataloader.dataset.baselines[1])
        median_loss = 0.0
        zero_loss = 0.0

        for epoch in range(epochs):

            epoch_len = []
            running_loss = []
            total_loss = []
            loss = []
            for i in range(self.n_nets):
                nets[i].train()
                epoch_len.append(0)
                running_loss.append(0.0)
                total_loss.append(0.0)
                optimizers[i].zero_grad()
                loss.append(None)
            for i, data in enumerate(dataloader, 0):
                inputs, resp = data[0].double(), data[1].double()
                for j in range(self.n_nets):
                    outputs = nets[j](inputs)
                    loss[j] = criterion(outputs.squeeze(), resp.squeeze())
                    running_loss[j] += loss[j].item()
                    if (i + 1) % 10 == 0:
                        loss[j] = total_loss[j] * 0.9 + loss[j] * 0.1
                        loss[j].backward()
                        optimizers[j].step()
                        optimizers[j].zero_grad()
                        total_loss[j] = 0.0
                    else:
                        total_loss[j] += loss[j]
                    epoch_len[j] += 1
            if total_loss[0] != 0.0:
                for j in range(self.n_nets):
                    loss[j] = total_loss[j] * 0.9 + loss[j] * 0.1
                    loss[j].backward()
                    optimizers[j].step()
                    optimizers[j].zero_grad()
            if self.validation:
                val_loss = []
                val_size = []
                for i in range(self.n_nets):
                    nets[i].eval()
                    val_loss.append(0.0)
                    val_size.append(0)

                with torch.no_grad():
                    for i, data in enumerate(self.val_loader, 0):
                        inputs, resp = data[0].double(), data[1].double()
                        for j in range(self.n_nets):
                            outputs = nets[j](inputs)
                            # out=outputs.squeeze().item()
                            # if out < min_out:
                            #     min_out=out
                            # elif out > max_out:
                            #     max_out = out
                            loss[j] = criterion(outputs.squeeze(), resp.squeeze())
                            val_loss[j] += loss[j].item()
                            val_size[j] += 1
                            if epoch == 0 and j == 0:
                                loss_median = criterion(median, resp.squeeze())
                                loss_zero = criterion(torch.tensor(0), resp.squeeze())
                                median_loss += loss_median.item()
                                zero_loss += loss_zero.item()
                # lg.info('min: %s, max: %s',min_out,max_out)
                # lg.info('e: %d | %s training_loss: %.10f', epoch + 1, val_size, (val_loss[] / val_size))
                if self.tensorboard:
                    # self.s_writer.add_scalar("Loss/Val", (val_loss / val_size), epoch + 1)

                    writerdict = {"median": (median_loss / val_size[0]),
                                  'zero': (zero_loss / val_size[0])
                                  }
                    for i in range(self.n_nets):
                        writerdict[nets[i].name + ": train"] = (running_loss[i] / epoch_len[0])
                        writerdict[nets[i].name + ": val"] = (val_loss[i] / val_size[0])
                        lg.info('%s e: %d | %s training_loss: %.10f',
                                str.ljust(nets[i].name, self.max_namesize, ' ')
                                , epoch + 1, epoch_len[0], (running_loss[i] / epoch_len[0]))

                        lg.info('%s e: %d | %s val_loss:      %.10f',
                                str.ljust(nets[i].name, self.max_namesize, ' ')
                                , epoch + 1, val_size[0], (val_loss[i] / val_size[0]))
                    self.s_writer.add_scalars("Loss/combined", writerdict, epoch + 1)
