import torch.onnx

from embed_nets import *

nets = [PoolingNetPlus(), Pooling_Net(), Mixed_Net(), Pre_net(), Pre_Net_Text(), CombinedAdaptivePool(1),
        PadPoolLayer(), PlusPoolLayer()]

print(torch.jit.trace(nets[0], (torch.ones((1, 28, 50)), torch.ones((1, 4, 500)))).graph)

for i, net in enumerate(nets):
    net.eval()
    data = (torch.ones((1, 28, 50)),)
    inputnames = ('input',)
    outputnames = ('output',)
    dynamic_axes = {'input': {0: 'batch', 2: 'sample_size'}, 'output': {0: 'batch'}}
    if i == 0:
        data = (torch.ones((1, 28, 50)), torch.ones((1, 4, 500)))
        inputnames = ('input_multi', 'input_text')
        dynamic_axes = {'input_multi': {0: 'batch', 2: 'sample_size'}, 'input_text': {0: 'batch', 2: 'sample_size'},
                        'output': {0: 'batch'}}

    if i == 4:
        data = (torch.ones((1, 4, 50)))
    if i == 5:
        dynamic_axes = {'input': {0: 'batch', 2: 'sample_size'}}
    if i == 7:
        data = (torch.ones((1, 28, 50)), torch.ones((1, 28, 500)))
        inputnames = ('input', 'project_dimension')
        dynamic_axes = {'input': {0: 'batch', 2: 'sample_size'}, 'output': {0: 'batch', 2: 'project_dimension'}}
    print(i)
    torch.onnx.export(net, data, "runs/vdl/" + net.name + ".onnx", export_params=False, input_names=inputnames,
                      output_names=outputnames, dynamic_axes=dynamic_axes, verbose=True)
