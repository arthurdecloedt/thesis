import logging as lg

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import embed_nets
import multiset

logFormatter = lg.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = lg.getLogger()
rootLogger.setLevel(lg.INFO)

net = embed_nets.PoolingNetPlus().double()
# net = embed_nets.Pooling_Net().double()

net.train()
# net2.train()
writer = SummaryWriter()
trainset = multiset.ContigSet()
ctv = trainset.contig_vals
for a in range(trainset.contig_vals.shape[-1]):
    col = ctv[:, a]
    vals = col.squeeze()
    # vals = np.log10(vals[vals != 0.])
    # vals_nan = np.isnan(vals)
    # vals_nan =  np.logical_not( vals_nan)
    # vals = vals[ vals_nan]
    d = np.abs(vals - np.median(vals))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    vals = vals[s < 6]
    vals = vals.squeeze()
    lg.info(vals.shape)
    counts, limits = np.histogram(vals, 20)
    sum_sq = vals.dot(vals)
    writer.add_histogram_raw(
        tag='vals_embed',
        min=vals.min(),
        max=vals.max(),
        num=len(vals),
        sum=vals.sum(),
        sum_squares=sum_sq,
        bucket_limits=limits[1:].tolist(),
        bucket_counts=counts.tolist(),
        global_step=a)
    writer.add_histogram_raw(
        tag='val_%s_embed' % a,
        min=vals.min(),
        max=vals.max(),
        num=len(vals),
        sum=vals.sum(),
        sum_squares=sum_sq,
        bucket_limits=limits[1:].tolist(),
        bucket_counts=counts.tolist())

writer.close()
