import csv
import datetime
import json
import logging as lg
import os
import warnings
from os.path import join

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import traceback


# this class is a small container with some parts of a multiset,
class ContigSet:

    def __init__(self, path='/data/leuven/332/vsc33219/data/Multiset') -> None:
        super().__init__()

        self.contig_vals = np.load(join(path, 'contig_vals.npy'), allow_pickle=True)
        self.contig_resp = np.load(join(path, 'contig_resp.npy'), allow_pickle=True)
        self.contig_usable = np.load(join(path, 'contig_usable.npy'), allow_pickle=True)
        self.has_contig = True

    def get_contig(self):
        assert self.has_contig
        return self.contig_vals[self.contig_usable], self.contig_resp[self.contig_usable]


# parse data from the dtai source file
def parse_moments(momentfile, rescale=False):
    momentsel = 4
    lg.info("Using dtai mvix30 data")

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


# parse data from cboe source, this will shift data so that we are predicting the next day iv
def parse_vix(aaplfile, rescale=False, shift=1, lvix=False, vixfile='', vix_shift=0,movement=False):
    vxaapl = pd.read_csv(aaplfile, index_col=0, usecols=[0, 4], parse_dates=True).dropna()
    lg.info("Using cboe vxAAPL data")

    dates = vxaapl.index.to_pydatetime()
    dates_np = np.array([np.datetime64(d.date()) for d in dates])
    vxaapl = vxaapl[vxaapl.columns[0]].to_numpy()

    # calculating changes in vix, using log for possible summability
    mult_change = vxaapl[shift:] / vxaapl[:-shift]
    mult_change = np.log(mult_change)
    mult_change_std = np.std(mult_change)
    mult_change_mean = np.nanmean(mult_change)
    classes = np.ones((vxaapl.shape[0],))
    classes = np.where(mult_change < mult_change_mean - mult_change_std/4,0,classes)
    classes = np.where(mult_change > mult_change_mean + mult_change_std/4,2,classes)
    classes_one_hot = np.zeros((classes.size, classes.max()+1))
    classes_one_hot[np.arange(classes.size),classes] = 1
    # we want to predict the vxaapl of the next day
    if lvix:
        vix = pd.read_csv(vixfile, index_col=0, usecols=[0, 4], parse_dates=True).dropna()

        # skip first vix date
        vdates = vix.index.to_pydatetime()
        vdates_np = np.array([np.datetime64(d.date()) for d in vdates])
        vix_array = vix[vix.columns[0]].to_numpy()

        # we cant just shift when whe have the vix, we also have to shift the vix back and make sure we have all values.
        i_dates, dind, vind = np.intersect1d(dates_np, vdates_np, return_indices=True)
        vind = vind[:-max(shift, vix_shift)] + vix_shift
        dind = dind[:-max(shift, vix_shift)] + shift
        i_dates = i_dates[:-max(shift, vix_shift)]
        vxaapl = vxaapl[dind]
        classes_one_hot = classes_one_hot[dind]
        vix_array = vix_array[vind]

        if rescale:
            min = np.min(vxaapl)
            vxaapl = vxaapl - min
            quant90 = np.quantile(vxaapl, 0.9)
            vxaapl = vxaapl / quant90
        else:
            min=0
            quant90 = 1
        lg.info("performed rescale: (vxaapl - %d ) / %d", min, quant90)

        lg.info("mean response: %s, median response: %s", np.mean(vxaapl), np.median(vxaapl))

        ret =(i_dates, vxaapl, 1, (np.mean(vxaapl), np.median(vxaapl)), vix_array, (min, quant90))

        return (*ret , classes_one_hot ) if movement else ret
    # vxaapldiff = vxaapl[1:] - vxaapl[:-1]
    else:
        dates_np = dates_np[:-shift]
        vxaapl = vxaapl[shift:]
        lg.info("mean response: %s, median response: %s", np.mean(vxaapl), np.median(vxaapl))
        min = np.min(vxaapl)
        vxaapl = vxaapl - min
        quant90 = np.quantile(vxaapl, 0.9)
        vxaapl = vxaapl / quant90
        lg.info("performed rescale: (vxaapl - %d ) / %d", min, quant90)
        lg.info("mean response rescaled : %s, median response: %s", np.mean(vxaapl), np.median(vxaapl))

        return dates_np, vxaapl, 1, (np.mean(vxaapl), np.median(vxaapl), (min, quant90))


# this is the main dataset class, altough it's been optimized in the plus class
# this class is still functional but is a lot slower than the plus class where the text embedding step has been
# multithreaded
class MultiSet(Dataset):

    def __init__(self, prefs, contig_resp=False,movement=True):
        self.movement = movement
        self._lreturns = False
        self.vix_arr = None
        self.vix = True
        self.contig_dates = np.array([])
        self.contig_resp = None
        self.has_contig = contig_resp
        self.prefs = prefs
        self.lookback=False
        self.n_l = 4
        preembedfolder = prefs['preembedfolder']
        jsonfile = prefs['jsonfile']
        lg.info('Starting initialization and preprocessing of multimodal dataset')
        lg.debug("using files %s and %s", preembedfolder, jsonfile)
        # We load the whole dataset, preembedded images and text this makes it a lot easier
        self.contig_ids = np.array([])
        self.contig_vals = np.array([])
        self.date_arr = np.array([])
        self.embedlen = 24
        self.analyser = SentimentIntensityAnalyzer()
        # arrays with id's of preembedded picturees
        self.norms = np.ones(self.embedlen)

        # we read the ANP classifications and embed the images into emotions
        self.init_imagefolder(preembedfolder)

        # this function takes the most time, it evaluates sentiment with vadersentiment
        self.read_json_aapl()

        # we only want days with enough dates
        if (True):
            vals, counts = np.unique(self.contig_dates, return_counts=True)
            ttl = np.sum(counts < 10)
            vals = vals[counts < 10]
            self.date_arr = self.date_arr[np.invert(np.isin(self.date_arr, vals))]
            lg.info("rejected %s dates for size (cutoff was 10)", ttl)
        # initialize array
        self.resp_arr = np.full(self.date_arr.shape, .64)

        # if we want the vix we have to parse it specially
        if self.vix:
            resp = parse_vix(self.prefs['vxaapl'], False, lvix=True, vixfile=self.prefs['vix'],movement=self.movement)
            response_dates, response, self.scale, self.baselines, vix_arr, self.scaling = resp[6:]
            _, d_inds, r_inds = np.intersect1d(self.date_arr, response_dates, return_indices=True)

            if self.movement:
                self.classes = np.zeros(self.resp_arr.shape)
                self.classes[d_inds] = resp[-1][r_inds]
            # we can only use days where we have both input and response (response might be shifted)
            self.vix_arr = np.full(self.date_arr.shape, -1.)

            self.vix_arr[d_inds] = vix_arr[r_inds]
        else:
            response_dates, response, self.scale, self.baselines, self.scaling = parse_vix(
                self.prefs['vxaapl'], False)
            # we can only use days where we have both input and response (response might be shifted)

            _, d_inds, r_inds = np.intersect1d(self.date_arr, response_dates, return_indices=True)
        self.resp_arr[d_inds] = response[r_inds]
        self.resp_inds = d_inds
        self.training_idx, self.test_idx = [], []
        # our dates will have to be sorted, timsort because they are almost sorted and default is quicksort
        self.date_id = np.argsort(self.contig_dates, kind='stable')
        self.contig_vals = np.ascontiguousarray(self.contig_vals[self.date_id])
        self.contig_dates = np.ascontiguousarray(self.contig_dates[self.date_id])

        # construct contig arrays if requested, this means this model can be used as contigset or save a contigset
        if self.has_contig:
            self.contig_resp = np.full(self.contig_dates.shape, -1.)
            self.contig_usable = np.full(self.contig_dates.shape, False)
            vals, c_inds, r_inds = np.intersect1d(self.contig_dates, response_dates, return_indices=True)

            for i in range(len(c_inds)):
                val = vals[i]
                cl = c_inds[i]
                cr = np.searchsorted(self.contig_dates, val, 'right')
                self.contig_resp[cl:cr] = response[r_inds[i]]
                self.contig_usable[cl:cr] = True

            print(self.contig_resp.shape)

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

    @property
    def lreturns(self):
        return self._lreturns

    @lreturns.setter
    def lreturns(self,val:bool):
        if val and not self.movement:
            lg.error("Tried to enable log return classification mode for dataset not initialized with these returns")
            raise ValueError("Tried to enable log return classification mode for dataset not initialized with these returns")
        self._lreturns = val

    # saves a contig version of a model
    def save_contig(self, path='/data/leuven/332/vsc33219/data/Multiset'):
        np.save(join(path, 'contig_vals.npy'), self.contig_vals)
        np.save(join(path, 'contig_resp.npy'), self.contig_resp)
        np.save(join(path, 'contig_usable.npy'), self.contig_usable)
        lg.info('saved contig array')

    # saves the whole model, old and had problems with casting
    def save(self, path='/data/leuven/332/vsc33219/data/Multiset'):
        warnings.warn("probs doesn't cover all func", DeprecationWarning)

        np.save(join(path, 'contig_vals.npy'), self.contig_vals)
        np.save(join(path, 'contig_dates.npy'), self.contig_dates)
        np.save(join(path, 'contig_ids.npy'), self.contig_ids)

        np.save(join(path, 'resp_inds.npy'), self.resp_inds)
        np.save(join(path, 'resp_arr.npy'), self.resp_arr)

        np.save(join(path, 'date_arr.npy'), self.date_arr)
        np.save(join(path, 'date_id.npy'), self.date_id)
        np.save(join(path, 'misc.npy'), [self.scale, self.embedlen])
        np.save(join(path, 'norms.npy'), self.norms)

    # loads a whole model, old and had problems with casting
    def unscale(self, y):
        return self.scaling[0] + (y * self.scaling[1])

    @classmethod
    def from_file(cls, path='/data/leuven/332/vsc33219/data/Multiset'):
        warnings.warn("probs doesn't cover all func", DeprecationWarning)

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

        with open('../resources/preferences.yaml') as f:
            obj.prefs = yaml.load(f, Loader=yaml.FullLoader)

        return obj

    # saves embedding to tensorboard projection feature
    def write_embedding(self, writer: SummaryWriter, n_rows=2000):

        row_inds = np.random.randint(0, self.contig_ids.shape[0], n_rows)
        rows = self.contig_vals[row_inds]
        rows_t = torch.from_numpy(rows)
        writer.add_embedding(rows_t, list(self.contig_dates[row_inds]), tag='Multiset: 24_mean_5 | 4_VADER')

    # load np files and embeds anp's to plutchik dyads
    # this function prob is due a reewrite, an multiprocessing shared mem ...
    # rewrite was not considered a rewrite because it's not really a bottleneck
    def init_imagefolder(self, preembedfolder):
        lg.info("starting preembed data processing")

        # the names of all files in the folder with the embeddings, the files where split because the total embedding
        # is rather big (on the order of gb's)
        names = [f.name for f in os.scandir(preembedfolder) if f.is_file()]
        lg.info("got %s files", len(names))
        preem_dict_ids = {}
        # tuples containing borders of id range
        preem_borders = {}
        # arrays of (processed) preembedded pictures
        preem_dict_vals = {}

        for n in names:

            # id's are saved as an int and thus in a different array and file
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

                # the classification vectors have to be embedded to be usefull
                preem_dict_vals[nr] = self.process_anps(vals)
                lg.debug('processed vals: %s', n)

            else:
                lg.warning('Unexpected file: %s', n)

        # the dict structure was the original one but its inefficient, we want the data in one contiguous slab
        lg.info("sorting and reallocating embedded data")

        # adding all the data together
        interm_contig_vals = preem_dict_vals['0']
        interm_contig_ids = preem_dict_ids['0']
        for key, val in preem_dict_vals.items():
            if key == '0':
                continue
            interm_contig_ids = np.concatenate((interm_contig_ids, preem_dict_ids[key]))
            interm_contig_vals = np.concatenate((interm_contig_vals, val), 0)
        # we want to sort along id's, this will mostly be done already therefore timsort
        inds = np.argsort(interm_contig_ids, kind="stable")
        n = inds.shape[0]
        interm_contig_ids = interm_contig_ids[inds]
        interm_contig_vals = interm_contig_vals[inds]
        # this was a wrong way to normalize data, which is not even really neccessary,
        # self.norms = np.linalg.norm(interm_contig_vals, axis=0)
        # interm_contig_vals = interm_contig_vals / self.norms

        # noinspection PyTypeChecker
        self.contig_dates = np.empty(n, dtype='datetime64[D]')

        self.contig_vals = np.concatenate((interm_contig_vals, np.zeros((n, 4))), 1)

        # we want to be sure that its one array for speed
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

    # this function takes ANP classification vectors
    # it embeds them in pluchik emotions according to an embedding by MVSO project
    def process_anps(self, vals):
        lg.debug("starting ANP data processing")
        top_nr = 7
        anp_file_n = self.prefs['anp_file_n']
        em_file_n = self.prefs['em_file_n']
        classes = np.array(json.load(open(anp_file_n)))
        ems = np.full((len(classes), self.embedlen), 1 / self.embedlen)

        # get emotional embedding for all ANP's
        with open(em_file_n, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            # skip first line
            next(reader)
            a = 0
            unfound=[]
            for row in reader:
                name = row[0]
                vals_loc = [float(f) for f in row[1:]]
                num_vals = np.array(vals_loc)
                loc = np.where(classes == name)[0]
                # check if ANP in class (some arent)
                if loc.shape != (0,):
                    ems[loc[0]] = num_vals
                # else:
                #     unfound.append(loc[0])
                # else:
                #     lg.warning("found a ANP emotional embedding (%s) not in classes nr %s", name,a)
                #     a+=1
        proc_val = np.zeros((len(vals), self.embedlen))
        lg.debug("loading ANP embeddings, using top %s ANPs", top_nr)
        # Convert anp classification into emotion embedding
        lg.debug(vals.shape)
        ind = -top_nr

        for n in range(len(vals)):
            # we select the top_nr highest scoring ANP's for this sample
            # assumption is that there are not a lot of unfound ANP's
            ind_n = ind
            while not np.any(proc_val[n]):
                top_inds = np.argpartition(vals[n], ind_n)[ind_n:]

                # we sum the value
                for ti in top_inds:
                    # if ti not in unfound:
                    proc_val[n] += ems[ti]
                ind_n += ind
            proc_val[n] = proc_val[n] / np.max(proc_val[n])
        return proc_val

    def __len__(self):
        # Here, we need to return the number of samples in this dataset.
        s = len(self.contig_ids)
        return s

    # this function returns the data for a certain date
    # param index: int, index of dat in date_arr
    def __getitem__(self, index):
        lg.debug("getting item: %s", index)
        date = self.date_arr[index]
        # lg.debug(date)
        # print(self.lookback)
        # traceback.print_stack()
        # we look for the left and side for this date to appear,
        # this is done by a binary search because the data is sorted
        ind_0 = np.searchsorted(self.contig_dates, date, side="left")
        ind_n = np.searchsorted(self.contig_dates, date, side='right')
        if ind_0 >= self.contig_vals.shape[0] or ind_n <= 0:
            raise ValueError('date not found')
        # here we reshape the data into the right tensor for our network
        # shape = (1 x 28 x sample_size)
        # first dim is batchsize but we have to keep this one because sample_size varies
        data = self.contig_vals[ind_0:ind_n]
        data = data.transpose((1, 0))
        data_tens = torch.from_numpy(data)
        resp_t = torch.tensor(self.resp_arr[index]) if not self._lreturns else torch.tensor(self.classes[index])
        if self.lookback:
            l = self.n_l
            resp_lb_n = np.zeros(l)
            if index == 0:
                pass
            elif index < l:
                resp_lb_n[-index:] = self.resp_arr[:index]
            else:
                resp_lb_n = self.resp_arr[index-l:index]
            for a in range(l):
                if resp_lb_n[a] <= 0:
                    resp_lb_n[a] = self.baselines[-1]
            resp_lb_t = torch.tensor(resp_lb_n)
            return data_tens, resp_t, torch.tensor(0) , resp_lb_t
        if self.vix:
            vix_t = torch.tensor(self.vix_arr[index])
            return data_tens, resp_t, vix_t

        return data_tens, resp_t, None

    # write embedding down for other embedding experiment
    def log_embedding(self, writer: SummaryWriter, dim=2000):

        assert self.has_contig
        inds = np.random.randint(20000,size=dim)

        vals = self.contig_vals[self.contig_usable][inds]
        resps = self.contig_resp[self.contig_usable][inds]

        bins = np.histogram_bin_edges(resps,25)
        resps = np.digitize(resps,bins)
        resps_s = list(resps)
        # vals = vals.transpose((1, 0))
        data_tens = torch.from_numpy(vals)

        writer.add_embedding(data_tens, global_step=4, tag='MVSO_top5_avg',metadata=resps_s)
        writer.flush()

    # get the contig data for this dataset, used in SK_Containers
    def get_contig(self):
        assert self.has_contig
        return self.contig_vals[self.contig_usable], self.contig_resp[self.contig_usable]

    @property
    def plus(self):
        return False

    # this function dominates time, and is optimized in plus model,
    # it reads tweets from a file, determines if they have an image in the set, if they do they are put in the dataset
    # mostly kept for future reference
    # the way this works is documented in the multiset_plus.py subclass, it works similar but is optimized
    def read_json_aapl(self, ):
        # {"_id":{"$oid":"5d7041dcd6c2261839ecf58f"},"username":"computer_hware","date":{
        # "$date":"2016-04-12T17:10:12.000Z"},"retweets":0,"favorites":0,"content":"#Apple iPhone SE release date, price,
        # specs and features: iPhone SE users report Bluetooth ... Read more: http://owler.us/aayzDR $ AAPL","geo":"",
        # "mentions":"","hashtags":"#Apple","replyTo":"","id":"719905464880726018",
        # "permalink":"https://twitter.com/computer_hware/status/719905464880726018"}
        contig_ids = self.contig_ids
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


# this is a wrapper around a multiset that divides its samples into bins of similar size
class Multi_Set_Binned(Dataset):

    @property
    def plus(self):
        return False

    def __init__(self, inner_set, reshuffle=False, ):
        super().__init__()
        self.training_idx, self.test_idx = None, None
        self.reshuffle = reshuffle
        lg.info("starting initialization binned dataset")
        self.inner = inner_set
        # bascially we want to split the datapoints as fairly as possible, and not into bins of 20
        # getting the inds and counts of unique dates
        lg.info('binning values')
        vals, inds, cnts = np.unique(self.inner.contig_dates, return_index=True, return_counts=True)
        # preallocating binn arr
        bins = np.full(self.inner.contig_dates.shape, -1, dtype='int64')
        # the number of bins for each unique date
        # we're underfilling when the count is not a multiple of 20 but above 30
        binsnr = np.asarray(np.ceil(np.true_divide(cnts, 20, out=np.ones(cnts.shape), where=cnts >= 30)), dtype='int64')
        # getting the the low binsize for all counts
        bnsize_u = np.asarray(np.floor_divide(cnts, binsnr), dtype='int64')
        # getting the amount of times the low value will be repeated for every date
        # the idea is that ( ubus_nr * bnsize_u ) + ((binsnr - ubus_nr ) * (bnsize +1) = cnts
        ubus_nr = (binsnr * (bnsize_u + 1)) - cnts
        counter = 0
        # if we want to reshuffle the bins correctly and efficiently we need this
        self.cnts = cnts
        self.inds = inds
        dates = np.zeros(np.sum(bnsize_u), dtype='int64')
        s_counter = 0
        b_scounter = 0
        for i in range(len(inds)):
            ind = np.searchsorted(self.inner.date_arr, vals[i])
            if ind < self.inner.resp_arr.shape[0] and self.inner.resp_arr[ind] != -1.:
                #  geting an array that expresses the sizes of the bins
                if bnsize_u[i] <= 5:
                    s_counter += 1
                    if cnts[i] >= 30:
                        lg.warning("date was binned with bins with size %s, count was %s", bnsize_u[i], cnts[i])
                    continue
                arr = np.repeat([bnsize_u[i], bnsize_u[i] + 1], [ubus_nr[i], binsnr[i] - ubus_nr[i]])
                # getting an array that expresses the bin nr for all dates in contig dates
                # print(arr)
                # print(binsnr[i])
                # print(cnts[i],bnsize_u[i],ubus_nr[i])
                bins[inds[i]:inds[i] + cnts[i]] = np.repeat(np.arange(counter, counter + binsnr[i]),
                                                            arr.astype('int64'))
                # here we record the date belonging to a bin for efficient retrieval later
                dates[counter:counter + binsnr[i]] = ind
                counter += binsnr[i]

        self.arrs = (binsnr, bnsize_u, ubus_nr, vals)
        self.bindates = np.ascontiguousarray(dates)
        self.bins = np.ascontiguousarray(bins)
        self.bins_u = np.unique(self.bins)
        self.c = counter
        lg.info('rejected %s dates/bins for size', s_counter)

    # this will shuffle the bins but the relatioin bins <-> dates will be kept
    def shuffle_bins(self):
        assert self.reshuffle
        for i in range(len(self.inds)):
            np.random.shuffle(self.bins[self.inds[i]:self.inds[i] + self.cnts[i]])

    # get an item, works very similar to the multiset one but checks for bins
    def __getitem__(self, index):
        if index == 0:
            return self.__getitem__(100)
        inner = self.inner
        lg.debug("getting item: %s", index)
        date = self.inner.date_arr[self.bindates[index]]
        # search for the dates to get a bounding box for data
        ind_0 = np.searchsorted(inner.contig_dates, date, side="left")
        ind_n = np.searchsorted(inner.contig_dates, date, side='right')
        if ind_0 >= inner.contig_vals.shape[0] or ind_n <= 0:
            raise ValueError('date not found')
        # check for samples in our bin
        sample_inds = np.squeeze(np.nonzero(self.bins[ind_0: ind_n] == index), axis=0)
        if len(sample_inds) == 0:
            print(index in self.bins_u)
            return None
        sample_inds += ind_0
        data = inner.contig_vals[sample_inds]
        data = data.transpose((1, 0))
        data_tens = torch.from_numpy(data)
        if self.inner.resp_arr[self.bindates[index]] == -1:
            lg.error("tried to get a responsevar -1")
        resp_t = torch.tensor(self.inner.resp_arr[self.bindates[index]])

        if self.inner.vix:
            vix_t = torch.tensor(self.inner.vix_arr[self.bindates[index]])
            return data_tens, resp_t, vix_t
        else:
            return data_tens, resp_t, 0

    @property
    def cr(self):
        return self.inner.has_contig

    @property
    def baselines(self):
        return self.inner.baselines

    @property
    def vix(self):
        return self.inner.baselines

    def __len__(self) -> int:
        return self.inner.__len__()

    def unscale(self, y):
        return self.inner.unscale(y)
