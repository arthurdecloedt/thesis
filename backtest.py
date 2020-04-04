import logging as lg


def backtest():
    pass





def train(container,epochs):
    net = container.net
    dataloader = container.dataloader
    optimizer = container.optimizer
    criterion = container.criterion

    for epoch in range(epochs):  # loop over the dataset epoch times
        running_loss = 0.0
        total_loss = 0.
        for i, data in enumerate(dataloader, 0):
            inputs, resp = data[0].double(), data[1].double()
            outputs = net(inputs)
            loss = criterion(outputs.squeeze(), resp.squeeze())

            if (i + 1) % 10 == 0:
                loss = total_loss * 0.9 + loss * 0.1
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss = 0.
            else:
                total_loss += loss

            running_loss += loss.item()
            if i % 100 == 99:
                lg.info('[%d, %5d] loss: %.3f',
                        epoch + 1, i + 1, running_loss / 100)
                running_loss = 0.0
