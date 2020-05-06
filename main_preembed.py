import imp
import sys

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataprocessing import *

MainModel = imp.load_source("MainModel", "./export_mvso.py")

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

batchsize = 500
dataloadr = DataLoader(dataset, batch_size=20, shuffle=False, sampler=sampler)
subdiv = 10000
# accnum = np.zeros((subdiv, 4342))
# accstr = np.empty((subdiv, 1), dtype=np.dtype(np.int64))
j = 0
n_partition = 0
writer = SummaryWriter()
a = iter(dataloadr)
d = next(a)
print(d)
print(d[0].shape)
writer.add_graph(the_model, d[0])
writer.flush()
# total = torch.zeros((0, 4342))
# for i, (images, ids) in enumerate(dataloadr):
#     predict = the_model(images)
#     if i == 0:
#         total = predict
#     else:
#         total = torch.cat((total, predict))
#     # predict_np_arr = predict.detach().numpy()
#     # idnp = np.array([[f] for f in ids])
#     # accnum[j * batchsize : (j + 1) * batchsize] = predict_np_arr
#     # accstr[j * batchsize : (j + 1) * batchsize] = idnp
#     # writer.add_embedding(predict,label_img=images,global_step=i,tag='MVSO_500')
#     j += 1
#     if j * (batchsize + 1) > subdiv:
#         writer.add_embedding(total, global_step=n_partition + 1, tag='MVSO_500')
#
#         # out_accstr= accstr[0 : (j + 1) * batchsize]
#         # out_accnum = accnum[0 : (j + 1) * batchsize]
#         break
#
# #         # save_partition(n_partition, out_accstr,out_accnum, output_folder)
# #         print('partition: ' + str(n_partition))
# #         print('batch nr: ' + str(i))
# #         print('sample nr: ' + str(i*batchsize))
# #         j = 0
# #         n_partition += 1
# # finalaccnum= accnum[0 : (j + 1) * batchsize]
# # finalaccstr= accstr[0 : (j + 1) * batchsize]
# # save_partition(n_partition, finalaccstr,finalaccnum, output_folder)
