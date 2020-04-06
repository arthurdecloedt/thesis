import logging as lg
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


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


class Pre_Net_Text(nn.Module):

    def __init__(self):
        super(Pre_Net_Text, self).__init__()

        super().__init__()
        self.c1 = nn.Conv1d(4, 10, 1)
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(28, 10)
        self.out = nn.Linear(10, 1)

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
        self.optimizer.zero_grad()
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
