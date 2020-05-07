import torch.onnx

from embed_nets import PoolingNetPlus

net = PoolingNetPlus()
net.eval()
data = (torch.ones((1, 28, 50)), torch.ones((1, 4, 500)))

torch.onnx.export(net, data, "model" + net.name + ".onnx")
