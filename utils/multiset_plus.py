import datetime
import json
import logging as lg
import multiprocessing as mp
import sys
import warnings
from multiprocessing import Pool
from multiprocessing.managers import SharedMemoryManager

import numpy as np
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils.multiset import MultiSet

analyser = None
shape_val_g = None
shape_id_g = None
ids_np_g = None
vals_np_g = None
dates_np_g = None
twt_date_np_g = None
twt_embed_np_g = None
shape_twt_dates_g = (600000,)
shape_twt_embed_g = (600000, 4)
multi_only = None


class MultiWrapper(MultiSet):

    def __init__(self, prefs, contig_resp=False):
        warnings.warn("You are instantiating this class without a combined multiset, this is not realy what you want")
        self.inner = MultiSetCombined(prefs, contig_resp)

    def __getattr__(self, attr):
        if attr not in self.__dict__:
            return getattr(self.inner, attr)
        return super().__getattr__(attr)

    @property
    def lookback(self):
        return self.inner.lookback

    @lookback.setter
    def lookback(self, val):
        self.inner.lookback=val

    @property
    def lreturns(self):
        return self.inner.lreturns

    @lreturns.setter
    def lreturns(self,val:bool):
        self.inner.lreturns=val


    def __getitem__(self, index):
        return self.inner._get_multi_only(index)

    @classmethod
    def construct(cls, dataset):
        assert isinstance(dataset, MultiSetCombined)
        obj = cls.__new__(cls)
        obj.inner = dataset
        return obj


class MultiSetPlus(MultiSet):
    def __init__(self, prefs, contig_resp=False, n_procs=30):
        self.n_procs = n_procs
        super().__init__(prefs, contig_resp)

    def read_json_aapl(self):
        # {"_id":{"$oid":"5d7041dcd6c2261839ecf58f"},"username":"computer_hware","date":{
        # "$date":"2016-04-12T17:10:12.000Z"},"retweets":0,"favorites":0,"content":"#Apple iPhone SE release date, price,
        # specs and features: iPhone SE users report Bluetooth ... Read more: http://owler.us/aayzDR $ AAPL","geo":"",
        # "mentions":"","hashtags":"#Apple","replyTo":"","id":"719905464880726018",
        # "permalink":"https://twitter.com/computer_hware/status/719905464880726018"}
        contig_ids = self.contig_ids

        # check if we are running in text/image pair only
        lg.info("starting json processing in combined mode")
        start_date = datetime.date(2011, 12, 29)
        end_date = datetime.date(2019, 9, 20)
        delta = end_date - start_date

        date_arr = [start_date + datetime.timedelta(days=i) for i in range(delta.days + 1)]

        date_arr_np = np.array([np.datetime64(d) for d in date_arr])

        n_tweets = 0
        n_fail = 0
        n_t_tweets = 0
        n_procs = self.n_procs
        d_arr = np.zeros(shape=contig_ids.shape, dtype=self.contig_dates.dtype)

        with open(self.prefs['jsonfile'], "r", encoding="utf-8") as file:
            with SharedMemoryManager() as smm:

                shape_id = self.contig_ids.shape
                shape_val = self.contig_vals.shape

                # allocating shared mem for the contig arrays we already have
                ids_np_sm = smm.SharedMemory(size=self.contig_ids.nbytes)
                vals_np_sm = smm.SharedMemory(size=self.contig_vals.nbytes)
                dates_sm = smm.SharedMemory(self.contig_dates.nbytes)

                # creating np array with shared mem as buffer
                ids_np = np.ndarray(shape_id, dtype=np.int64, buffer=ids_np_sm.buf)
                vals_np = np.ndarray(shape_val, buffer=vals_np_sm.buf)
                dates_np = np.ndarray(self.contig_dates.shape, dtype=self.contig_dates.dtype, buffer=dates_sm.buf)

                # copying previous infomration into shared arrays
                np.copyto(ids_np, self.contig_ids)
                np.copyto(vals_np, self.contig_vals)
                np.copyto(dates_np, self.contig_dates)
                with Pool(n_procs, setup_subprocess_plus, (vals_np_sm, ids_np_sm, dates_sm, shape_id, shape_val,
                                                           self.contig_dates.shape)) as embed_pool:
                    lg.info("initialized pool of %s processes, %s  extra processes are running according to mp",
                            n_procs, len(mp.active_children()))
                    lg.info("one process extra is expected for the shared memory manager")
                    for p in mp.active_children():
                        lg.debug(p.name)

                    results = embed_pool.imap_unordered(embed_line, filegenerator(file, n_t_tweets),
                                                        chunksize=5 * n_procs)
                    for id, found, date in results:
                        n_t_tweets += 1
                        if found:
                            # no search over array -> duplicates but we can do this more efficiently later
                            # search here is non parallel
                            d_arr[n_tweets] = date
                            n_tweets += 1
                # copying contig values and dates back we want its new state
                np.copyto(self.contig_vals, vals_np)
                np.copyto(self.contig_dates, dates_np)
                un = np.unique(d_arr)
                # copying the embedded tweets into array
        dates = date_arr_np.shape[0]
        d_arr = np.intersect1d(date_arr_np, un)
        n_null = dates - d_arr.shape[0]
        lg.info("collected %s tweets with %s errors", n_tweets, n_fail)
        lg.info("collected tweets on %s dates", (delta.days - n_null))
        lg.info("deleted %s dates", n_null)
        self.date_arr = d_arr


class MultiSetCombined(MultiSet):

    @property
    def plus(self):
        return True

    def __init__(self, prefs, contig_resp=False, n_procs=30):
        self.contig_t_embed = None
        self.contig_t_dates = None
        self.n_procs = n_procs
        self.orig = False
        super().__init__(prefs, contig_resp)

    def _get_multi_only(self, index):
        return super().__getitem__(index)

    def __getitem__(self, index):
        x = super().__getitem__(index)
        date = self.date_arr[index]
        ind_0 = np.searchsorted(self.contig_t_dates, date, side="left")
        ind_n = np.searchsorted(self.contig_t_dates, date, side='right')
        if ind_0 >= self.contig_t_embed.shape[0] or ind_n <= 0:
            raise ValueError('date not found')
        data = self.contig_t_embed[ind_0:ind_n]
        data = data.transpose((1, 0))
        data_tens = torch.from_numpy(data)
        return *x, data_tens

    def read_json_aapl(self):
        # {"_id":{"$oid":"5d7041dcd6c2261839ecf58f"},"username":"computer_hware","date":{
        # "$date":"2016-04-12T17:10:12.000Z"},"retweets":0,"favorites":0,"content":"#Apple iPhone SE release date, price,
        # specs and features: iPhone SE users report Bluetooth ... Read more: http://owler.us/aayzDR $ AAPL","geo":"",
        # "mentions":"","hashtags":"#Apple","replyTo":"","id":"719905464880726018",
        # "permalink":"https://twitter.com/computer_hware/status/719905464880726018"}
        contig_ids = self.contig_ids


       # check if we are running in text/image pair only
        lg.info("starting json processing in combined mode")
        start_date = datetime.date(2011, 12, 29)
        end_date = datetime.date(2019, 9, 20)
        delta = end_date - start_date

        date_arr = [start_date + datetime.timedelta(days=i) for i in range(delta.days + 1)]

        date_arr_np = np.array([np.datetime64(d) for d in date_arr])

        n_tweets = 0
        n_fail = 0
        n_t_tweets = 0
        n_procs = self.n_procs
        d_arr = np.zeros(shape=contig_ids.shape, dtype=self.contig_dates.dtype)

        with open(self.prefs['jsonfile'], "r", encoding="utf-8") as file:
            with SharedMemoryManager() as smm:

                shape_id = self.contig_ids.shape
                shape_val = self.contig_vals.shape

                # allocating shared mem for the embedded values of the text tweets
                twt_dates_sm = smm.SharedMemory(size=np.empty(shape_twt_dates_g, dtype=self.contig_dates.dtype).nbytes)
                twt_embed_sm = smm.SharedMemory(size=np.empty(shape_twt_embed_g).nbytes)

                # allocating shared mem for the contig arrays we already have
                ids_np_sm = smm.SharedMemory(size=self.contig_ids.nbytes)
                vals_np_sm = smm.SharedMemory(size=self.contig_vals.nbytes)
                dates_sm = smm.SharedMemory(self.contig_dates.nbytes)

                # creating np array with shared mem as buffer
                ids_np = np.ndarray(shape_id, dtype=np.int64, buffer=ids_np_sm.buf)
                vals_np = np.ndarray(shape_val, buffer=vals_np_sm.buf)
                dates_np = np.ndarray(self.contig_dates.shape, dtype=self.contig_dates.dtype, buffer=dates_sm.buf)

                # copying previous infomration into shared arrays
                np.copyto(ids_np, self.contig_ids)
                np.copyto(vals_np, self.contig_vals)
                np.copyto(dates_np, self.contig_dates)
                with Pool(n_procs, setup_subproces_all, (
                        twt_dates_sm, twt_embed_sm, vals_np_sm, ids_np_sm, dates_sm, shape_id, shape_val,
                        self.contig_dates.shape)) as embed_pool:
                    lg.info("initialized pool of %s processes, %s  extra processes are running according to mp",
                            n_procs, len(mp.active_children()))
                    lg.debug("one process extra is expected for the shared memory manager")
                    for p in mp.active_children():
                        lg.debug(p.name)

                    results = embed_pool.imap_unordered(embed_line, filegenerator(file, n_t_tweets),
                                                        chunksize=5 * n_procs)
                    for id, found, date in results:
                        n_t_tweets += 1
                        if found:
                            # no search over array -> duplicates but we can do this more efficiently later
                            # search here is non parallel
                            d_arr[n_tweets] = date
                            n_tweets += 1
                # copying contig values and dates back we want its new state
                np.copyto(self.contig_vals, vals_np)
                np.copyto(self.contig_dates, dates_np)
                un = np.unique(d_arr)
                # copying the embedded tweets into array
                twt_dates_np = np.ndarray(shape=shape_twt_dates_g, dtype=self.contig_dates.dtype,
                                          buffer=twt_dates_sm.buf)
                twt_embed_np = np.ndarray(shape=shape_twt_embed_g, buffer=twt_embed_sm.buf)
                self.contig_t_dates = np.copy(twt_dates_np)
                self.contig_t_embed = np.copy(twt_embed_np)

        dates = date_arr_np.shape[0]
        d_arr = np.intersect1d(date_arr_np, un)
        n_null = dates - d_arr.shape[0]

        t_inds = np.isin(self.contig_t_dates, date_arr_np)
        # we want timsort coz almost sorted
        t_sort = np.argsort(self.contig_t_dates[t_inds], kind="stable")

        td_interm = self.contig_t_dates[t_inds]
        te_interm = self.contig_t_embed[t_inds]
        self.contig_t_dates = np.ascontiguousarray(td_interm[t_sort])
        self.contig_t_embed = np.ascontiguousarray(te_interm[t_sort])

        lg.info("collected %s tweets with %s errors", n_tweets, n_fail)
        lg.info("collected tweets on %s dates", (delta.days - n_null))
        lg.info("deleted %s dates", n_null)
        self.date_arr = d_arr


def setup_subproces_all(contig_tw_date_sm, contig_tw_embed_sm, contig_vals_sm, contig_id_sm, contig_date_sm, shape_id,
                        shape_val, shape_dates):
    setup_subprocess_plus(contig_vals_sm, contig_id_sm, contig_date_sm, shape_id,
                          shape_val, shape_dates)
    global twt_date_np_g, twt_embed_np_g, shape_twt_dates_g, shape_twt_embed_g, multi_only

    twt_date_np_g = np.ndarray(shape=shape_twt_dates_g, dtype='datetime64[D]', buffer=contig_tw_date_sm.buf)
    twt_embed_np_g = np.ndarray(shape=shape_twt_embed_g, buffer=contig_tw_embed_sm.buf)
    multi_only = False


def setup_subprocess_plus(contig_vals_sm, contig_id_sm, contig_date_sm, shape_id,
                          shape_val, shape_dates):
    global shape_val_g, shape_id_g, ids_np_g, vals_np_g, dates_np_g, analyser, multi_only
    shape_id_g = shape_id
    shape_val_g = shape_val
    analyser = SentimentIntensityAnalyzer()

    ids_np_g = np.ndarray(shape=shape_id_g, dtype=np.int64, buffer=contig_id_sm.buf)
    vals_np_g = np.ndarray(shape=shape_val_g, buffer=contig_vals_sm.buf)
    dates_np_g = np.ndarray(shape=shape_dates, dtype='datetime64[D]', buffer=contig_date_sm.buf)
    multi_only = True


def embed_line(arg):
    line, n = arg
    global ids_np_g, vals_np_g, dates_np_g, multi_only
    try:
        jline = json.loads(line)
        datestr = jline['date']['$date']
        datet = datetime.datetime.strptime(datestr, "%Y-%m-%dT%H:%M:%S.%fZ")
        id = int(jline['id'])
        content = jline['content']
        date = datet.date()
        loc = ids_np_g.searchsorted(id)
        n_date = np.datetime64(date)
        if not multi_only:
            global twt_date_np_g, twt_embed_np_g
            sent = analyser.polarity_scores(content)
            arr = [sent['neg'], sent['neu'], sent['pos'], sent['compound']]
            nparr = np.array(arr)
            twt_embed_np_g[n] = nparr
            twt_date_np_g[n] = n_date

            if loc < ids_np_g.shape[0] and id == ids_np_g[loc]:
                vals_np_g[loc, 24:] = arr
                dates_np_g[loc] = n_date
                return id, True, n_date
            return id, False, None
        else:
            if loc < ids_np_g.shape[0] and id == ids_np_g[loc]:
                sent = analyser.polarity_scores(content)
                arr = [sent['neg'], sent['neu'], sent['pos'], sent['compound']]
                nparr = np.array(arr)
                vals_np_g[loc, 24:] = nparr
                dates_np_g[loc] = n_date
                return id, True, n_date
            return id, False, None


    except Exception as e:
        print(e)
        sys.stdout.flush()
        raise e



def filegenerator(file, n, max=None):
    l = file.readline()
    if max is None:
        while l:
            yield l, n
            l = file.readline()
            n += 1
    else:
        while l and n < max:
            yield l, n
            l = file.readline()
            n += 1
