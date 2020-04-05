import logging as lg

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
    def __init__(self, net, dataloader, optimizer, criterion, validation=False, val_loader=None):
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.optimizer.zero_grad()
        self.net = net
        self.validation = validation
        assert (not validation) or val_loader is not None
        self.val_loader = val_loader

    def train(self, epochs):
        net = self.net
        dataloader = self.dataloader
        optimizer = self.optimizer
        criterion = self.criterion

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
            optimizer.zero_grad()
            if not self.validation:
                continue
            net.eval()
            val_loss = 0.0
            val_size = 0
            min_out = 5.0
            max_out = 0.0
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
            # lg.info('min: %s, max: %s',min_out,max_out)
            lg.info('e: %d | %s val_loss: %.10f', epoch + 1, val_size, (val_loss / val_size))
