import multiprocessing
from itertools import cycle
from multiprocessing.pool import Pool

import imageParser
import image_downloader
import json_parser
from queue import *

file_name = "../data/aapl.json"
proxy_file_nm ="../data/http_proxies.txt"
with open(proxy_file_nm,"r") as proxy_file:
    proxies =proxy_file.read().splitlines()



if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    manager = multiprocessing.Manager()
    url_queue = manager.Queue(60000)

    proxy_queue = manager.Queue(60000)
    for p in proxies:
        proxy_queue.put(p)

    trash_queue = manager.Queue(60000)
    img_queue = manager.Queue(60000)
    pJson = multiprocessing.Process(target=json_parser.get_tweet_queue,args=[file_name,url_queue])

    pJson.start()
    pool = Pool(2)

    # start first url before entering loop
    counter = 0
    url = url_queue.get(block=True)
    pool.apply_async(imageParser.enqueue_image_url_proxy, (url,proxy_queue,trash_queue))
    n = 260000
    while not url_queue.empty():
        url = url_queue.get(block=True)
        # a new url needs to be processed
        n+=1
        counter += 1
        ap_as = pool.apply_async(imageParser.enqueue_image_url_proxy, (url,proxy_queue,trash_queue))
        if counter >= 1000:
            ap_as.get()
            print('scrapes:{0}'.format(str(n)))
            counter=0
    pool.close()
    pJson.terminate()
    pool.join()
   while not proxy_queue.empty():
      x=0
       proxy_queue.get()
        x+=1
    print(x)
    x=0
    while not trash_queue.empty():
        trash_queue.get()
        x+=1
    print(x)