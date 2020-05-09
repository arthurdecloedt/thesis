import multiprocessing as mp
import os
import time
from multiprocessing.managers import SharedMemoryManager

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = None
a = 10
ctd = None
shape_val = None
shape_id = None
ids_np = None
vals_np = None


def initialize_pool(val_sh, id_shape, val_sm, ids_sm):
    shape_id = id_shape
    shape_val = val_sh
    global analyser
    analyser = SentimentIntensityAnalyzer()
    global ids_np
    ids_np = np.ndarray(shape=shape_id, buffer=ids_sm.buf)
    global vals_np
    vals_np = np.ndarray(shape=shape_val, buffer=val_sm.buf)
    print(__name__, 'init')


def target(i):
    # global ids_np, vals_np
    # vals_np[i] = np.arange(i, i + 28)
    print(i)
    global a

    time.sleep(2)
    return __name__, os.getpid(), a


if __name__ == '__main__':
    mp.set_start_method('spawn')
    print('test')
    shape_id = (1000,)
    shape_val = (1000, 28)

    ids = np.arange(1000)
    vals = np.random.uniform(0, 5, shape_val)
    with SharedMemoryManager() as smm:
        a = 9
        ids_sm = smm.SharedMemory(ids.nbytes)
        vals_sm = smm.SharedMemory(vals.nbytes)
        ids_np = np.ndarray(shape=shape_id, buffer=ids_sm.buf)
        np.copyto(ids_np, ids)
        # initialize_pool, (shape_val, shape_id, vals_sm, ids_sm)
        with mp.Pool(5, initialize_pool, (shape_val, shape_id, vals_sm, ids_sm)) as pool:
            arr = [(i, 10 - i) for i in range(10)]

            for p in mp.active_children():
                print(p.name)
            res = pool.imap_unordered(target, arr)
            for a in res:
                print(a)
            ctd = np.copy(np.ndarray(shape=shape_val, buffer=vals_sm.buf))
