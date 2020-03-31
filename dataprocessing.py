import datetime
import json
import os

import numpy as np
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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


def save_partition(n_partition, accstr, accnum, output):
    np.save(output + 'array_' + str(n_partition) + '_ids.npy', accstr)
    np.save(output + 'array_' + str(n_partition) + '_vals.npy', accnum)
    print('saved: ' + output + 'array_' + str(n_partition) + '.npy')


class multi_set(Dataset):

    def __init__(self, preembedfolder, jsonfile):

        # We load the whole dataset, preembedded images and text this makes it a lot easier

        self.analyser = SentimentIntensityAnalyzer()
        # arrays with id's of preembedded picturees
        self.preem_dict_ids = {}
        # arrays of preembedded pictures
        self.preem_dict_vals = {}
        # tuples containing borders of id range
        self.preem_borders={}
        self.preembedfolder = preembedfolder
        names = [f.name for f in os.scandir(preembedfolder) if f.is_file()]
        for n in names:
            if n.contains('ids'):
                s = n.split('_')
                nr = s[1]
                ids = np.load(n)
                self.preem_dict_ids[nr] = ids
                assert ids[0] < ids[-1]
                self.preem_borders[nr] = (ids[0],ids[-1])
            elif n.contains('vals'):
                s = n.split('_')
                nr = s[1]
                vals= np.load(n)
                self.preem_dict_vals[nr] = vals
            else:
                print('Unexpected file: ' + n)
        self.datedict = readjson(jsonfile)

    def __len__(self):
        # Here, we need to return the number of samples in this dataset.
        s = 0
        for a in self.preem_dict_ids.values():
            s += a.size
        return s

    def __getitem__(self, index):

        sample = self.datedict[index]

        for (id,content) in sample:
            vadersent = self.analyser.polarity_scores(content)



def readjson(jsonfile, img_id_dict=None, img_b_dict=None):
    # {"_id":{"$oid":"5d7041dcd6c2261839ecf58f"},"username":"computer_hware","date":{
    # "$date":"2016-04-12T17:10:12.000Z"},"retweets":0,"favorites":0,"content":"#Apple iPhone SE release date, price,
    # specs and features: iPhone SE users report Bluetooth ... Read more: http://owler.us/aayzDR $ AAPL","geo":"",
    # "mentions":"","hashtags":"#Apple","replyTo":"","id":"719905464880726018",
    # "permalink":"https://twitter.com/computer_hware/status/719905464880726018"}
    only_img=False
    if not (img_b_dict is None or img_id_dict is None) :
        only_img=True
    start_date = datetime.date(2012, 12, 31)
    end_date = datetime.date(2019, 9, 5)
    delta = start_date - end_date

    date_arr = [start_date + datetime.timedelta(days=i) for i in range(delta.days + 1)]
    date_arr_np = np.array(date_arr)
    date_dict = dict.fromkeys(date_arr, [])
    n_tweets = 0
    n_fail = 0
    skipped = 0

    with open(jsonfile, "r", encoding="utf-8") as file:
        line = file.readline()
        line = True
        while line:
            try:
                line = file.readline()
                jline = json.loads(line)
                datestr = jline['date'][['$date']]
                datet = datetime.datetime.strptime(datestr, "%Y-%m-%dT%H:%M:%S.%fZ")
                id = int(jline['id'])
                content = jline[content]
                date = datet.date()
                if only_img:
                    for i in img_b_dict.keys():
                        (lowerb,upperb) = img_b_dict[i]
                        if lowerb <= id <= upperb:
                            if id in img_b_dict[i]:
                                date_dict[date].append((id, content))
                                n_tweets += 1
                                break
                    skipped+=1
                    continue
                else:
                    date_dict[date].append((id, content))
                    n_tweets += 1

            except Exception as e:
                print(str(e))
                n_fail += 1
        n_null = 0

        for date in date_arr:
            if not date_dict[date]:
                date_arr_np = date_arr_np[date_arr_np != date]
                n_null += 1
    print("collected %s tweets with %s errors" % (n_tweets, n_fail))
    print("collected tweets on %s dates" % (delta.days - n_null))

    return date_arr_np, date_dict
