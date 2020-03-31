import importlib

import torch
import imp
from torch.utils.data import DataLoader
import sys

from dataprocessing import *

MainModel = imp.load_source("MainModel", "./export.py")


weight_file = sys.argv[1]
ims_location = sys.argv[2]
output_folder = sys.argv[3]


# weight_file= '/data/leuven/332/vsc33219/data/deepsentibank/export'
#
# ims_location = '/data/leuven/332/vsc33219/data/full'
# output_folder= '/data/leuven/332/vsc33219/data/preembedtest/'

the_model = torch.load(weight_file)
the_model.eval()

dataset = ImageDataSet(ims_location)
sampler = IdSampler(dataset)

batchsize = 200
dataloadr = DataLoader(dataset, batch_size=batchsize, shuffle=False, sampler=sampler)
subdiv = 5000
accnum = np.zeros((subdiv, 2089))
accstr = np.empty((subdiv, 1), dtype=np.dtype(np.int64))
j = 0
n_partition = 0

for i, (images, ids) in enumerate(dataloadr):
    predict = the_model(images)
    predict_np_arr = predict.detach().numpy()
    idnp = np.array([[f] for f in ids])
    accnum[j * batchsize : (j + 1) * batchsize] = predict_np_arr
    accstr[j * batchsize : (j + 1) * batchsize] = idnp
    j += 1
    if j * (batchsize + 1) > subdiv:
        out_accstr= accstr[0 : (j + 1) * batchsize]
        out_accnum = accnum[0 : (j + 1) * batchsize]
        save_partition(n_partition, out_accstr,out_accnum, output_folder)
        print('partition: ' + str(n_partition))
        print('batch nr: ' + str(i))
        print('sample nr: ' + str(i*batchsize))
        j = 0
        n_partition += 1
finalaccnum= accnum[0 : (j + 1) * batchsize]
finalaccstr= accstr[0 : (j + 1) * batchsize]
save_partition(n_partition, finalaccstr,finalaccnum, output_folder)
