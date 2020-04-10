import csv
import datetime
import json
import logging as lg
import os
from os.path import join

import numpy as np
import pandas as pd
import torch
import yaml
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


def parse_moments(momentfile, rescale=False):
    momentsel = 4

    moments = pd.read_csv(momentfile, index_col=0, usecols=[0, momentsel], parse_dates=True).dropna()

    dates = moments.index.to_pydatetime()
    dates_np = np.array([np.datetime64(d.date()) for d in dates])
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
        self.contig_dates = None
        self.prefs = prefs
        preembedfolder = prefs['preembedfolder']
        jsonfile = prefs['jsonfile']
        lg.info('Starting initialization and preprocessing of multimodal dataset')
        lg.debug("using files %s and %s", preembedfolder, jsonfile)
        # We load the whole dataset, preembedded images and text this makes it a lot easier
        self.contig_ids = None
        self.contig_vals = None
        self.date_arr = None
        self.embedlen = 24
        self.analyser = SentimentIntensityAnalyzer()
        # arrays with id's of preembedded picturees
        self.norms = np.ones(self.embedlen)

        self.init_imagefolder(preembedfolder)

        self.read_json_aapl()

        self.resp_arr = np.empty(self.date_arr.shape)
        response_dates, response, self.scale, self.baselines = parse_moments(self.prefs['moments'], True)

        _, d_inds, r_inds = np.intersect1d(self.date_arr, response_dates, return_indices=True)

        self.resp_arr[d_inds] = response[r_inds]
        self.resp_inds = d_inds
        self.training_idx, self.test_idx = [], []
        self.date_id = np.argsort(self.contig_dates)

        self.contig_vals = np.ascontiguousarray(self.contig_vals[self.date_id])
        self.contig_dates = np.ascontiguousarray(self.contig_dates[self.date_id])
        assert np.all(self.contig_dates[:-1] <= self.contig_dates[1:])
        lg.info("finished initializing the dataset")

    def create_valsplit(self, distr=0.8):
        l = self.date_arr.shape[0]
        n_train = int(l * distr)
        indices = np.random.permutation(l)
        self.training_idx, self.test_idx = indices[:n_train], indices[n_train:]

    def create_temporal_valsplit(self, distr=0.8):
        l = self.resp_inds.shape[0]
        n_train = int(l * distr)
        indices = np.arange(l)
        self.training_idx, self.test_idx = indices[:n_train], indices[n_train:]

    def save(self, path='/data/leuven/332/vsc33219/data/Multiset'):

        np.save(join(path, 'contig_vals.npy'), self.contig_vals)
        np.save(join(path, 'contig_dates.npy'), self.contig_dates)
        np.save(join(path, 'contig_ids.npy'), self.contig_ids)

        np.save(join(path, 'resp_inds.npy'), self.resp_inds)
        np.save(join(path, 'resp_arr.npy'), self.resp_arr)

        np.save(join(path, 'date_arr.npy'), self.date_arr)
        np.save(join(path, 'date_id.npy'), self.date_id)
        np.save(join(path, 'misc.npy'), [self.scale, self.embedlen])
        np.save(join(path, 'norms.npy'), self.norms)

    @classmethod
    def from_file(cls, path='/data/leuven/332/vsc33219/data/Multiset'):

        obj = cls.__new__(cls)
        super(MultiSet, obj).__init__()
        obj.contig_vals = np.load(join(path, 'contig_vals.npy'), allow_pickle=True)
        obj.contig_dates = np.load(join(path, 'contig_dates.npy'), allow_pickle=True)
        obj.contig_ids = np.load(join(path, 'contig_ids.npy'), allow_pickle=True)

        obj.resp_inds = np.load(join(path, 'resp_inds.npy'), allow_pickle=True)
        obj.resp_arr = np.load(join(path, 'resp_arr.npy'), allow_pickle=True)

        obj.date_arr = np.load(join(path, 'date_arr.npy'), allow_pickle=True)
        obj.date_id = np.load(join(path, 'date_id.npy'), allow_pickle=True)

        obj.norms = np.load(join(path, 'norms.npy'), allow_pickle=True)

        a = np.load(join(path, 'misc.npy'), allow_pickle=True)
        obj.scale = a[0]
        obj.embedlen = a[1]

        mean = np.mean(obj.resp_arr[np.nonzero(obj.resp_arr)])
        median = np.median(obj.resp_arr[np.nonzero(obj.resp_arr)])

        obj.baselines = (mean, median)

        obj.analyser = SentimentIntensityAnalyzer()

        with open('resources/preferences.yaml') as f:
            obj.prefs = yaml.load(f, Loader=yaml.FullLoader)

        return obj

    def init_imagefolder(self, preembedfolder):
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
        n = inds.shape[0]
        interm_contig_ids = interm_contig_ids[inds]
        interm_contig_vals = interm_contig_vals[inds]
        self.norms = np.linalg.norm(interm_contig_vals, axis=0)
        interm_contig_vals = interm_contig_vals / self.norms

        self.contig_dates = np.empty(n, dtype='datetime64[D]')

        self.contig_vals = np.concatenate((interm_contig_vals, np.zeros((n, 4))), 1)
        self.contig_ids = np.ascontiguousarray(interm_contig_ids)
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
        ind_0 = np.searchsorted(self.contig_dates, date, side="left")
        ind_n = np.searchsorted(self.contig_dates, date, side='right')
        if ind_0 >= self.contig_vals.shape[0] or ind_n <= 0:
            raise ValueError('date not found')
        data = self.contig_vals[ind_0:ind_n]
        data = data.transpose((1, 0))
        data_tens = torch.from_numpy(data)
        resp_t = torch.tensor(self.resp_arr[index])
        return data_tens, resp_t

    def read_json_aapl(self, ):
        # {"_id":{"$oid":"5d7041dcd6c2261839ecf58f"},"username":"computer_hware","date":{
        # "$date":"2016-04-12T17:10:12.000Z"},"retweets":0,"favorites":0,"content":"#Apple iPhone SE release date, price,
        # specs and features: iPhone SE users report Bluetooth ... Read more: http://owler.us/aayzDR $ AAPL","geo":"",
        # "mentions":"","hashtags":"#Apple","replyTo":"","id":"719905464880726018",
        # "permalink":"https://twitter.com/computer_hware/status/719905464880726018"}
        contig_ids = self.contig_ids
        # check if we are running in text/image pair only
        lg.info("starting json processing in only_img mode")
        start_date = datetime.date(2011, 12, 29)
        end_date = datetime.date(2019, 9, 20)
        delta = end_date - start_date

        date_arr = [start_date + datetime.timedelta(days=i) for i in range(delta.days + 1)]

        date_arr_np = np.array([np.datetime64(d) for d in date_arr])
        boolarr = np.zeros(date_arr_np.shape, dtype=np.bool_)
        n_tweets = 0
        n_fail = 0
        skipped = 0

        with open(self.prefs['jsonfile'], "r", encoding="utf-8") as file:
            line = file.readline()
            while line:
                try:
                    jline = json.loads(line)
                    datestr = jline['date']['$date']
                    datet = datetime.datetime.strptime(datestr, "%Y-%m-%dT%H:%M:%S.%fZ")
                    id = int(jline['id'])
                    content = jline['content']
                    date = datet.date()
                    loc = contig_ids.searchsorted(id)
                    if loc < contig_ids.shape[0] and id == contig_ids[loc]:
                        sent = self.analyser.polarity_scores(content)
                        arr = np.array([sent['neg'], sent['neu'], sent['pos'], sent['compound']])
                        self.contig_vals[loc, 24:] = arr
                        n_date = np.datetime64(date)
                        self.contig_dates[loc] = n_date
                        boolarr[np.argwhere(date_arr_np == n_date)] = True
                        n_tweets += 1
                    skipped += 1

                except Exception as e:
                    lg.error(line)
                    n_fail += 1
                    raise e
                finally:
                    line = file.readline()
        dates = date_arr_np.shape[0]
        pos_inds = np.argwhere(boolarr).squeeze()
        date_arr_np = date_arr_np[pos_inds]
        n_null = dates - date_arr_np.shape[0]

        lg.info("collected %s tweets with %s errors", n_tweets, n_fail)
        lg.info("collected tweets on %s dates", (delta.days - n_null))
        lg.info("deleted %s dates", n_null)
        self.date_arr = date_arr_np


class MultiSampler(torch.utils.data.Sampler):
    # data_source: MultiSet
    def __init__(self, data_source):
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


class Multi_Set_Cst(MultiSet):
    def __init__(self, prefs):
        super().__init__(prefs)

        # ok so this is a bit convoluted
        # bascially we want to split the datapoints as fairly as possible, and not into bins of 20
        # getting the inds and counts of unique dates
        vals, inds, cnts = np.unique(self.contig_dates, return_index=True, return_counts=True)
        # preallocating binn arr
        bins = np.zeros(self.contig_dates.shape, dtype='int64')
        # the number of bins for each unique date
        # we're underfilling when the count is not a multiple of 20 but above 30
        binsnr = np.ceil(np.true_divide(cnts, 20, out=np.ones(cnts.shape), where=cnts >= 30))
        # getting the the low binsize for all counts
        bnsize_u = np.floor_divide(cnts, binsnr)
        # getting the amount of thimes the low value will be repeated for every date
        # the idea is that ( ubus_nr * bnsize_u ) + ((binsnr - ubus_nr ) * (bnsize +1) = cnts
        ubus_nr = (cnts - (binsnr * (bnsize_u + 1))) / (2 * bnsize_u - 1)
        counter = 0

        dates = np.empty(np.sum(bnsize_u), dtype='datetime64')
        for i in range(len(inds)):
            #  geting an array that expresses the sizes of the bins
            arr = np.repeat([bnsize_u[i], bnsize_u[i] + 1], [ubus_nr[i], cnts[i] - ubus_nr[i]])
            # getting an array that expresses the bin nr for all dates in contig dates
            bins[inds[i]:inds[i] + cnts[i]] = np.random.permutation(
                np.repeat(np.arange(counter, counter + binsnr[i]), arr.astype('int64')))
            # here we record the date belonging to a bin for efficient retrieval later
            dates[counter:counter + binsnr[i]] = vals[i]
            counter += binsnr[i]
        self.bindates = np.ascontiguousarray(dates)
        self.bins = np.ascontiguousarray(bins)
