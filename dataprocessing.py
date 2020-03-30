import os
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



class ImageDataSet(Dataset):
    def __init__(self, root='train', transform=transforms.ToTensor()):
        self.root = root
        self.transform = transform
        self.paths = [f.path for f in os.scandir(root) if f.is_file()]
        names = [f.name for f in os.scandir(root) if f.is_file()]

        self.ids = [f.split('.')[0] for f in names]

        self.ids.sort()

    def __len__(self):
        # Here, we need to return the number of samples in this dataset.
        return len(self.paths)

    def __getitem__(self, index):
        img_name = os.path.join(self.root, index + '.jpg')
        image = Image.open(img_name)
        image = image.resize((224, 224))
        if self.transform is not None:
            image = self.transform(image)

        return image, index


class IdSampler(Sampler):

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(self.data_source.ids)

    def __len__(self):
        return len(self.data_source)


def save_partition(n_partition, accumulator, output):
    np.save(output + 'array_' + str(n_partition) + '.npy', accumulator)
    print('saved: ' + output + 'array_' + str(n_partition) + '.npy')

