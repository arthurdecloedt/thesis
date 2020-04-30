import logging as lg

import numpy as np
import torch
# from PIL import Image
# from PIL import ImageFile
from torch.utils.data import Dataset, Sampler

from multiset import MultiSet, Multi_Set_Binned


# from torch import transforms

# ImageFile.LOAD_TRUNCATED_IMAGES = True
#
#
# class ImageDataSet(Dataset):
#     def __init__(self, root='train', transform=transforms.ToTensor()):
#         self.root = root
#         self.transform = transform
#         self.paths = [f.path for f in os.scandir(root) if f.is_file()]
#         names = [f.name for f in os.scandir(root) if f.is_file()]
#
#         self.ids = [f.split('.')[0] for f in names]
#
#         self.ids.sort()
#         lg.debug("dataset input_side initialized")
#
#     def __len__(self):
#         # Here, we need to return the number of samples in this dataset.
#         return len(self.paths)
#
#     def __getitem__(self, index):
#         img_name = os.path.join(self.root, index + '.jpg')
#         image = Image.open(img_name)
#         image = image.resize((224, 224))
#         if self.transform is not None:
#             image = self.transform(image)
#
#         return image, index


class IdSampler(Sampler):

    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        return iter(self.data_source.ids)

    def __len__(self):
        return len(self.data_source)


def save_partition(n_partition, accstr, accnum, output):
    np.save(output + 'array_' + str(n_partition) + '_ids.npy', accstr)
    np.save(output + 'array_' + str(n_partition) + '_vals.npy', accnum)
    lg.debug('saved: ' + output + 'array_' + str(n_partition) + '.npy')


class MultiSampler(torch.utils.data.Sampler):
    data_source: MultiSet

    def __init__(self, data_source: MultiSet):
        super().__init__(data_source)
        self.data_source = data_source
        self.inds = data_source.resp_inds
        self.l = len(self.inds)
        lg.info("sampler inititiated with %s samples", len(self))

    def __iter__(self):
        return iter(np.random.permutation(self.inds))

    def __len__(self):
        return self.l


class MultiSplitSampler(MultiSampler):
    def __init__(self, data_source, train=True):
        super().__init__(data_source)
        inds = self.data_source.training_idx if train else self.data_source.test_idx
        self.inds = np.intersect1d(self.inds, inds)
        self.l = len(self.inds)
        lg.info("split sampler inititiated with %s samples", len(self))

    def __iter__(self):
        return iter(np.random.permutation(self.inds))


class MultiBinSampler(Sampler):

    def __init__(self, data_source: Multi_Set_Binned):
        super().__init__(data_source)
        lg.info("Binsampler supplied with %s bins", data_source.c)
        self.data_source = data_source
        self.inds = np.arange(data_source.c)
        self.train = True
        self.len = data_source.c

    def get_val_sampler(self, distr=.8, temporal=True):

        if temporal:
            _, dinds = np.unique(self.data_source.bindates[self.data_source.bindates != 0], return_index=True)
            d_l = len(dinds)
            cutoff = int(d_l * distr)
            self.inds = np.arange(dinds[cutoff])
            val_inds = np.arange(dinds[cutoff], self.data_source.c)
            val_sampler = MultiBinSampler.__new__(MultiBinSampler)
            val_sampler.inds = val_inds
            val_sampler.data_source = self.data_source
            val_sampler.train = False
            val_sampler.len = self.data_source.c - dinds[cutoff]
            lg.info("Binsampler split to %s training samples and %s validation samples", dinds[cutoff],
                    self.data_source.c - dinds[cutoff])
            self.len = dinds[cutoff]
            return val_sampler
        raise NotImplementedError

    def __iter__(self):
        if self.data_source.reshuffle and self.train:
            self.data_source.shuffle_bins()
        return iter(np.random.permutation(self.inds))

    def __len__(self):
        return self.len
