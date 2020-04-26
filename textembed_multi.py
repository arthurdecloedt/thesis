import datetime
import json
import logging as lg
import sys
import traceback
import warnings
from multiprocessing import Pool
from pathos.multiprocessing import ProcessPool

# from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager

import ctypes

import yaml
import numpy as np
# noinspection PyUnresolvedReferences

from multiprocessing.sharedctypes import RawArray
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from multiset import MultiSet

analyser = SentimentIntensityAnalyzer()

ctd = None
def embed_line(line,n,contig_t_d,contig_t_embed):
    try:
        if line == '':
            return (-1,None,[])
        contig_t_embed_local=np.ndarray((600000,4), dtype=np.float, buffer=contig_t_embed.buf)
        contig_t_d_local=np.ndarray((600000,), dtype=np.float, buffer=contig_t_d.buf)
        jline = json.loads(line)
        datestr = jline['date']['$date']
        datet = datetime.datetime.strptime(datestr, "%Y-%m-%dT%H:%M:%S.%fZ")
        id = int(jline['id'])
        content = jline['content']
        date = datet.date()
        sent = analyser.polarity_scores(content)
        arr = [sent['neg'], sent['neu'], sent['pos'], sent['compound']]
        nparr = np.array(arr)
        contig_t_embed_local[n]=nparr
        contig_t_d_local[n]=datet.toordinal()
        n_date = np.datetime64(date)
        return id, n_date, arr
    except Exception as e:
        lg.error(str(e))
        raise e


if __name__ == '__main__':
    import time

    start_time = time.time()

    # {"_id":{"$oid":"5d7041dcd6c2261839ecf58f"},"username":"computer_hware","date":{
    # "$date":"2016-04-12T17:10:12.000Z"},"retweets":0,"favorites":0,"content":"#Apple iPhone SE release date, price,
    # specs and features: iPhone SE users report Bluetooth ... Read more: http://owler.us/aayzDR $ AAPL","geo":"",
    # "mentions":"","hashtags":"#Apple","replyTo":"","id":"719905464880726018",
    # "permalink":"https://twitter.com/computer_hware/status/719905464880726018"}
    with open('resources/preferences.yaml') as f:
        prefs = yaml.load(f, Loader=yaml.FullLoader)

    # multiset = MultiSet(prefs)

    with SharedMemoryManager() as smm:
        # contig_ids = multiset.contig_ids
        # noinspection PyUnresolvedReferences
        contig_t_embed = smm.SharedMemory(size=np.empty((600000,4)).nbytes)
        contig_t_d= smm.SharedMemory(size=np.empty((600000,4),dtype='int64').nbytes)

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
        batch = 500
        n_procs = 10
        with open(prefs['jsonfile'], "r", encoding="utf-8") as file:
            with Pool(n_procs) as embed_pool:
                lines = [(file.readline(), n_t_tweets + a, contig_t_d, contig_t_embed) for a in range(batch)]
                # lines = [file.readline() for a in range(batch)]
                # ctd_s = [contig_t_d for a in range(batch)]
                # cte_s = [contig_t_embed for a in range(batch)]
                # n = range(n_t_tweets,n_t_tweets+batch)
                # locs = range(n_t_tweets,n_t_tweets + batch)
                while lines[-1][0]:
                    try:
                        # results = embed_pool.uimap(embed_line,lines,n,ctd_s,cte_s)
                        results = embed_pool.imap_unordered(embed_line,lines)
                        for id, n_date, arr in results:
                            # loc = contig_ids.searchsorted(id)
                            # if loc < contig_ids.shape[0] and id == contig_ids[loc]:
                            #     multiset.contig_vals[loc, 24:] = arr
                            #     multiset.contig_dates[loc] = n_date
                            #     boolarr[np.argwhere(date_arr_np == n_date)] = True
                            n_t_tweets += 1

                            n_tweets += 1
                    except Exception as e:
                        lg.error(e)
                        n_fail += 1
                        raise e
                    finally:
                        lines = [(file.readline(), n_t_tweets + a, contig_t_d, contig_t_embed) for a in range(batch)]
                        # lines = [file.readline() for a in range(batch)]
                        # ctd_s = [contig_t_d for a in range(batch)]
                        # cte_s = [contig_t_embed for a in range(batch)]
                        # n = range(n_t_tweets, n_t_tweets + batch)
                        if n_t_tweets >= 100000:
                            print(time.time() - start_time)
                            print(n_t_tweets)

                            break


        # dates = date_arr_np.shape[0]
        # pos_inds = np.argwhere(boolarr).squeeze()
        # date_arr_np = date_arr_np[pos_inds]
        # n_null = dates - date_arr_np.shape[0]
        # lg.info("collected %s tweets with %s errors", n_tweets, n_fail)
        # lg.info("collected tweets on %s dates", (delta.days - n_null))
        # lg.info("deleted %s dates", n_null)
        print(time.time()-start_time)
        ctd = np.ndarray((600000,4),np.float,buffer=contig_t_embed.buf)
