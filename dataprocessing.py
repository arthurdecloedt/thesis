import csv
import datetime
import json
import logging as lg
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
        lg.debug("dataset initialized")

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
    lg.debug('saved: ' + output + 'array_' + str(n_partition) + '.npy')


class MultiSet(Dataset):

    def __init__(self, preembedfolder, jsonfile):
        lg.debug('Starting initialization and preprocessing of multimodal dataset')
        lg.info("using files %s and %s", preembedfolder, jsonfile)
        # We load the whole dataset, preembedded images and text this makes it a lot easier
        self.preem_dict_ids = {}
        # tuples containing borders of id range
        self.preem_borders = {}
        # arrays of (processed) preembedded pictures

        self.preem_dict_vals = {}

        self.embedlen = 24
        self.analyser = SentimentIntensityAnalyzer()
        # arrays with id's of preembedded picturees
        self.init_imagefolder(preembedfolder)
        self.datedict = readjson(jsonfile)
        lg.debug("finished initializing the dataset")

    def init_imagefolder(self, preembedfolder):
        lg.debug("starting preembed data processing")
        names = [f.name for f in os.scandir(preembedfolder) if f.is_file()]
        lg.info("got %s files", len(names))
        for n in names:
            if 'ids' in n:
                s = n.split('_')
                nr = s[1]
                file = os.path.join(preembedfolder, n)
                ids = np.load(file)
                self.preem_dict_ids[nr] = ids
                assert ids[0] < ids[-1]
                self.preem_borders[nr] = (ids[0], ids[-1])
                lg.info('processed ids: %s', n)
            elif 'vals' in n:
                s = n.split('_')
                nr = s[1]
                file = os.path.join(preembedfolder, n)
                vals = np.load(file)
                self.preem_dict_vals[nr] = self.process_anps(vals)
                lg.info('processed vals: %s', n)

            else:
                lg.warning('Unexpected file: %s', n)

    def process_anps(self, vals):
        lg.info("starting ANP data processing")

        top_nr = 5
        anp_file_n = '/data/leuven/332/vsc33219/data/english_label.json'
        em_file_n = '/data/leuven/332/vsc33219/data/ANP_emotion_mapping_english.csv'
        classes = np.array(json.load(open(anp_file_n)))
        ems = np.zeros((len(classes), self.embedlen))

        # get emotional embedding for all ANP's
        with open(em_file_n, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
                name = row[0]
                vals_loc = [float(f) for f in row[1:]]
                num_vals = np.array(vals_loc)
                loc = np.where(classes == name)[0]
                # check if ANP in class (some arent)
                if loc.shape != (0,):
                    ems[loc[0]] = num_vals
        proc_val = np.zeros((len(vals), self.embedlen))
        lg.info("loading ANP embeddings, using top %s ANPs", top_nr)
        # Convert anp classification into emotion embedding
        lg.debug(vals.shape)
        for n in range(len(vals)):
            ind = -top_nr
            top_inds = np.argpartition(vals[n], ind)[ind:]

            # get avg val for all emotions
            for ind in top_inds:
                proc_val[n] += ems[ind] / top_nr
        return proc_val

    def __len__(self):
        # Here, we need to return the number of samples in this dataset.
        s = 0
        for a in self.preem_dict_ids.values():
            s += a.size
        return s

    def __getitem__(self, index):
        lg.info("getting item: %s", index)
        sample = self.datedict[index]

        for (img_id, content) in sample:
            sent = self.analyser.polarity_scores(content)
            arr = np.empty(self.embedlen + 4)
            arr[:4] = [sent['neg'], sent['neu'], sent['pos'], sent['compound']]

            for i in self.preem_borders.keys():
                (lowerb, upperb) = self.preem_borders[i]
                if lowerb <= img_id <= upperb:
                    arr[4:] = self.preem_dict_vals[i][self.preem_dict_ids[i].searchsorted(img_id)]
                    break


def readjson(jsonfile, img_id_dict=None, img_b_dict=None):
    # {"_id":{"$oid":"5d7041dcd6c2261839ecf58f"},"username":"computer_hware","date":{
    # "$date":"2016-04-12T17:10:12.000Z"},"retweets":0,"favorites":0,"content":"#Apple iPhone SE release date, price,
    # specs and features: iPhone SE users report Bluetooth ... Read more: http://owler.us/aayzDR $ AAPL","geo":"",
    # "mentions":"","hashtags":"#Apple","replyTo":"","id":"719905464880726018",
    # "permalink":"https://twitter.com/computer_hware/status/719905464880726018"}
    only_img = False

    # check if we are running in text/image pair only
    if not (img_b_dict is None or img_id_dict is None):
        only_img = True
    start_date = datetime.date(2012, 12, 29)
    end_date = datetime.date(2019, 9, 20)
    delta = end_date - start_date

    date_arr = [start_date + datetime.timedelta(days=i) for i in range(delta.days + 1)]
    date_arr_np = np.array(date_arr)
    date_dict = dict.fromkeys(date_arr)
    for k, _ in date_dict.items():
        date_dict[k] = []
    d = 0
    test = []
    n_tweets = 0
    n_fail = 0
    skipped = 0
    with open(jsonfile, "r", encoding="utf-8") as file:
        line = file.readline()
        while line:
            try:
                jline = json.loads(line)
                datestr = jline['date']['$date']
                datet = datetime.datetime.strptime(datestr, "%Y-%m-%dT%H:%M:%S.%fZ")
                id = int(jline['id'])
                content = jline['content']
                date = datet.date()
                if not str(date) in test:
                    test.append(str(date))
                if only_img:
                    for i in img_b_dict.keys():
                        (lowerb, upperb) = img_b_dict[i]
                        if lowerb <= id <= upperb:
                            loc = img_b_dict[i].searchsorted[id]
                            if id == loc:
                                date_dict[date].append((id, content))
                                n_tweets += 1
                                break
                    skipped += 1
                else:
                    if not date_dict[date]:
                        d += 1
                    date_dict[date].append((id, content))
                    n_tweets += 1

            except Exception as e:
                lg.error(line)
                n_fail += 1
                raise e
            finally:
                line = file.readline()
        n_null = 0

        for date in date_arr:
            if not date_dict[date]:
                date_arr_np = date_arr_np[date_arr_np != date]
                n_null += 1
    lg.debug("collected %s dates ", d)
    lg.debug("collected %s tweets with %s errors", n_tweets, n_fail)
    lg.debug("collected tweets on %s dates", (delta.days - n_null))
    lg.debug("deleted %s dates", n_null)
    return date_arr_np, date_dict, test
