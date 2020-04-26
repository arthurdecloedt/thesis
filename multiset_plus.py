import datetime
import json
import logging as lg
from multiprocessing import Pool

import ctypes
# from pathos.multiprocessing import ProcessPool
import numpy as np
# noinspection PyUnresolvedReferences

from multiprocessing.sharedctypes import RawArray
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from multiset import MultiSet

analyser = SentimentIntensityAnalyzer()


class MultiSetCombined(MultiSet):

    def __init__(self, prefs, contig_resp=False):
        self.contig_t_embed = None
        self.contig_t_dates = None
        super().__init__(prefs, contig_resp)

    def read_json_aapl(self):
        # {"_id":{"$oid":"5d7041dcd6c2261839ecf58f"},"username":"computer_hware","date":{
        # "$date":"2016-04-12T17:10:12.000Z"},"retweets":0,"favorites":0,"content":"#Apple iPhone SE release date, price,
        # specs and features: iPhone SE users report Bluetooth ... Read more: http://owler.us/aayzDR $ AAPL","geo":"",
        # "mentions":"","hashtags":"#Apple","replyTo":"","id":"719905464880726018",
        # "permalink":"https://twitter.com/computer_hware/status/719905464880726018"}
        contig_ids = self.contig_ids

        contig_t_d_shared = RawArray(ctypes.c_long, 600000)
        contig_t_embed_shared = RawArray(ctypes.c_double, (600000,4))
        contig_t_d = np.ctypeslib.as_array(contig_t_d_shared,(600000,))
        contig_t_embed = np.ctypeslib.as_array(contig_t_embed_shared,(600000,4))

       # check if we are running in text/image pair only
        lg.info("starting json processing in combined mode")
        start_date = datetime.date(2011, 12, 29)
        end_date = datetime.date(2019, 9, 20)
        delta = end_date - start_date

        date_arr = [start_date + datetime.timedelta(days=i) for i in range(delta.days + 1)]

        date_arr_np = np.array([np.datetime64(d) for d in date_arr])

        boolarr = np.zeros(date_arr_np.shape, dtype=np.bool_)
        n_tweets = 0
        n_fail = 0
        skipped = 0
        n_t_tweets = 0
        batch = 100
        n_procs = 20
        with open(self.prefs['jsonfile'], "r", encoding="utf-8") as file:
            with ProcessPool(n_procs) as embed_pool:
                lines = [file.readline() for a in range(batch)]
                ctd_s = [contig_t_d_shared for a in range(batch)]
                cte_s = [contig_t_embed_shared for a in range(batch)]
                locs = range(n_t_tweets,n_t_tweets + batch)
                while lines[-1][0]:
                    try:
                        results = embed_pool.uimap(embed_line,lines,ctd_s,cte_s,locs)
                        for id, n_date, arr in results:
                            loc = contig_ids.searchsorted(id)
                            n_t_tweets += batch
                            if loc < contig_ids.shape[0] and id == contig_ids[loc]:
                                self.contig_vals[loc, 24:] = arr
                                self.contig_dates[loc] = n_date
                                boolarr[np.argwhere(date_arr_np == n_date)] = True
                                n_tweets += 1
                    except Exception as e:
                        lg.error(e)
                        n_fail += 1
                        raise e
                    finally:
                        line = file.readline()
        dates = date_arr_np.shape[0]
        pos_inds = np.argwhere(boolarr).squeeze()
        date_arr_np = date_arr_np[pos_inds]
        n_null = dates - date_arr_np.shape[0]
        self.contig_t_embed = contig_t_embed
        self.contig_t_dates = contig_t_d
        lg.info("collected %s tweets with %s errors", n_tweets, n_fail)
        lg.info("collected tweets on %s dates", (delta.days - n_null))
        lg.info("deleted %s dates", n_null)
        self.date_arr = date_arr_np


def embed_line(line,contig_t_d_shared,contig_t_embed_shared,n):
    try:
        contig_t_d = np.ctypeslib.as_array(contig_t_d_shared, (600000,))
        contig_t_embed = np.ctypeslib.as_array(contig_t_embed_shared, (600000, 4))
        jline = json.loads(line)
        datestr = jline['date']['$date']
        datet = datetime.datetime.strptime(datestr, "%Y-%m-%dT%H:%M:%S.%fZ")
        id = int(jline['id'])
        content = jline['content']
        date = datet.date()
        sent = analyser.polarity_scores(content)
        arr = [sent['neg'], sent['neu'], sent['pos'], sent['compound']]
        nparr = np.array(arr)
        contig_t_embed[n]=nparr
        contig_t_d[n]=datet.toordinal()
        n_date = np.datetime64(date)
        return id, n_date, arr
    except Exception as e:
        lg.error(str(e))
        raise e

