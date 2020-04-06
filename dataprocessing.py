import csv
import datetime
import json
import logging as lg
import os

import numpy as np
import pandas as pd
import torch
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
        lg.debug("dataset input_side initialized")

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


def parse_moments(momentfile, rescale=False):
    momentsel = 4

    moments = pd.read_csv(momentfile, index_col=0, usecols=[0, momentsel], parse_dates=True).dropna()

    dates = moments.index.to_pydatetime()
    dates_np = np.array([d.date() for d in dates])
    mom_array = moments[moments.columns[0]].to_numpy()

    scale = 1
    if rescale:
        scale = np.amax(mom_array)
        mom_array /= scale

    mean = np.mean(mom_array)
    median = np.median(mom_array)
    lg.info("mean response: %s, median response: %s", mean, median)
    return dates_np, mom_array, scale, (mean, median)


class MultiSet(Dataset):

    def __init__(self, prefs):
        self.prefs = prefs
        preembedfolder = prefs['preembedfolder']
        jsonfile = prefs['jsonfile']
        lg.info('Starting initialization and preprocessing of multimodal dataset')
        lg.debug("using files %s and %s", preembedfolder, jsonfile)
        # We load the whole dataset, preembedded images and text this makes it a lot easier
        self.contig_ids = None
        self.contig_vals = None

        self.embedlen = 24
        self.analyser = SentimentIntensityAnalyzer()
        # arrays with id's of preembedded picturees
        self.norms = np.ones(24)

        self.init_imagefolder(preembedfolder)

        self.date_arr, self.datedict = read_json_aapl(jsonfile, self.contig_ids)

        self.response_dates, self.response, self.scale, self.baselines = parse_moments(self.prefs['moments'], True)
        self.training_idx, self.test_idx = [], []
        lg.info("finished initializing the dataset")

    def create_valsplit(self, distr=0.8):
        l = self.date_arr.shape[0]
        n_train = int(l * distr)
        indices = np.random.permutation(l)
        self.training_idx, self.test_idx = indices[:n_train], indices[n_train:]

    def init_imagefolder(self, preembedfolder, contig=True):
        lg.info("starting preembed data processing")
        names = [f.name for f in os.scandir(preembedfolder) if f.is_file()]
        lg.info("got %s files", len(names))
        preem_dict_ids = {}
        # tuples containing borders of id range
        preem_borders = {}
        # arrays of (processed) preembedded pictures
        preem_dict_vals = {}

        for n in names:
            if 'ids' in n:
                s = n.split('_')
                nr = s[1]
                file = os.path.join(preembedfolder, n)
                ids = np.squeeze(np.load(file))

                preem_dict_ids[nr] = ids

                preem_borders[nr] = (ids[0], ids[-1])
                lg.debug('processed ids: %s', n)
            elif 'vals' in n:
                s = n.split('_')
                nr = s[1]
                file = os.path.join(preembedfolder, n)
                vals = np.load(file)

                preem_dict_vals[nr] = self.process_anps(vals)
                lg.debug('processed vals: %s', n)

            else:
                lg.warning('Unexpected file: %s', n)
        lg.info("sorting and reallocating embedded data")
        interm_contig_vals = preem_dict_vals['0']
        interm_contig_ids = preem_dict_ids['0']
        for key, val in preem_dict_vals.items():
            if key == '0':
                continue
            interm_contig_ids = np.concatenate((interm_contig_ids, preem_dict_ids[key]))
            interm_contig_vals = np.concatenate((interm_contig_vals, val), 0)
        inds = np.argsort(interm_contig_ids)
        interm_contig_ids = interm_contig_ids[inds]
        interm_contig_vals = interm_contig_vals[inds]
        self.norms = np.linalg.norm(interm_contig_vals, axis=0)
        interm_contig_vals = interm_contig_vals / self.norms
        self.contig_ids = np.ascontiguousarray(interm_contig_ids)
        self.contig_vals = np.ascontiguousarray(interm_contig_vals)
        lg.info("reallocation done ")

        # for u in unsorted:
        #     inds = np.argsort(self.preem_dict_ids[u])
        #     lg.debug('resorting array nr %s',u)
        #     lg.debug('orig borders: %s ', self.preem_borders[u])
        #     self.preem_dict_ids[u] = self.preem_dict_ids[u][inds]
        #     self.preem_dict_vals[u] = self.preem_dict_vals[u][inds]
        #     self.preem_borders[u] = (self.preem_dict_ids[u][0], self.preem_dict_ids[u][-1])
        #     lg.debug('new borders: %s',self.preem_borders[u])

    def process_anps(self, vals):
        lg.debug("starting ANP data processing")
        top_nr = 5
        anp_file_n = self.prefs['anp_file_n']
        em_file_n = self.prefs['em_file_n']
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
        lg.debug("loading ANP embeddings, using top %s ANPs", top_nr)
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
        len(self.contig_ids)
        return s

    def __getitem__(self, index):
        lg.debug("getting item: %s", index)
        date = self.date_arr[index]
        # lg.debug(date)
        sample = self.datedict[date]
        data = np.empty((len(sample), self.embedlen + 4))
        j = 0
        for (img_id, content) in sample:
            sent = self.analyser.polarity_scores(content)
            arr = np.zeros(self.embedlen + 4)
            arr[:4] = [sent['neg'], sent['neu'], sent['pos'], sent['compound']]

            ind = self.contig_ids.searchsorted(img_id)
            if ind < self.contig_ids.shape[0] and self.contig_ids[ind] == img_id:
                arr[4:] = self.contig_vals[index]
                data[j] = arr
                j += 1
            else:
                lg.warning("tried to load tweet without img")

        data = data[:j]
        data = data.transpose((1, 0))
        data_tens = torch.from_numpy(data)
        resp_i = np.where(self.response_dates == date)
        if len(resp_i) != 1:
            lg.error("tried to load date without response var (or multiple ones)")
            return data_tens, torch.tensor(0)
        resp_t = torch.tensor(self.response[resp_i])
        return data_tens, resp_t


class MultiSampler(torch.utils.data.Sampler):
    # data_source: MultiSet

    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source
        arr = self.data_source.date_arr
        lg.debug("sampler got %s dates inputside", len(arr))

        rarr = self.data_source.response_dates
        lg.info("sampler got %s dates responseside", len(rarr))

        _, self.inds, _ = np.intersect1d(arr, rarr, return_indices=True)
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


def read_json_aapl(jsonfile, contig_ids):
    # {"_id":{"$oid":"5d7041dcd6c2261839ecf58f"},"username":"computer_hware","date":{
    # "$date":"2016-04-12T17:10:12.000Z"},"retweets":0,"favorites":0,"content":"#Apple iPhone SE release date, price,
    # specs and features: iPhone SE users report Bluetooth ... Read more: http://owler.us/aayzDR $ AAPL","geo":"",
    # "mentions":"","hashtags":"#Apple","replyTo":"","id":"719905464880726018",
    # "permalink":"https://twitter.com/computer_hware/status/719905464880726018"}
    only_img = False

    # check if we are running in text/image pair only
    if not (contig_ids is None):
        only_img = True
        lg.info("starting json processing in only_img mode")
    else:
        lg.info("starting json processing in full mode")
    start_date = datetime.date(2011, 12, 29)
    end_date = datetime.date(2019, 9, 20)
    delta = end_date - start_date

    date_arr = [start_date + datetime.timedelta(days=i) for i in range(delta.days + 1)]
    date_arr_np = np.array(date_arr)
    date_dict = dict.fromkeys(date_arr)
    for k, _ in date_dict.items():
        date_dict[k] = []
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
                if only_img:

                    loc = contig_ids.searchsorted(id)
                    if loc < contig_ids.shape[0] and id == contig_ids[loc]:
                        date_dict[date].append((id, content))
                        n_tweets += 1
                    skipped += 1
                else:
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
                date_dict.pop(date)
                n_null += 1
    lg.info("collected %s tweets with %s errors", n_tweets, n_fail)
    lg.info("collected tweets on %s dates", (delta.days - n_null))
    lg.info("deleted %s dates", n_null)
    return date_arr_np, date_dict
