import multiprocessing
from multiprocessing.pool import Pool

import imageParser
import image_downloader
import json_parser
from queue import *
from pathlib import Path
import yappi
import sys

file_name = Path("data/TSLA.json")

if __name__ == '__main__':

    manager = multiprocessing.Manager()
    url_queue = manager.Queue(60000)

    proxy_queue = manager.Queue(60000)
    trash_queue = manager.Queue(60000)
    img_queue = manager.Queue(60000)
    pJson = multiprocessing.Process(target=json_parser.get_tweet_queue,args=[file_name,url_queue])

    pJson.start()
    
    pool = Pool(100)

    # start first url before entering loop
    counter = multiprocessing.Value('i', 0)
    c_lock = multiprocessing.Lock()
    url = url_queue.get(block=True)
    
    ap_as = pool.apply_async(imageParser.enqueue_image_url,(url,))
    n = 0

    ofset = 460000
    for a in range(ofset):
        url = url_queue.get(block=True)
    c = 0
    n=ofset
        
    while not (url_queue.empty() and n > ofset+5000):
        url = url_queue.get(block=True)
        # a new url needs to be processed
        n+=1
        c+=1
        ap_as = pool.apply_async(imageParser.enqueue_image_url, (url,))
        if c >= 1000:
            sys.stdout.write('\n')
            sys.stdout.flush()
            print('scheduled:{0}'.format(str(n)))
            for a in range(20):
                ap_as.wait(60)
                if ap_as.ready():
                    break   
                sys.stdout.write('\n')
                sys.stdout.flush()
                print("ran %s minutes already after completed submit" % a)
                
            c = 0
            sys.stdout.write('\n')
            sys.stdout.flush()
            print('performed:{0}'.format(str(n)))

    print('terminated')
    ap_as.get(240)
    pool.close()
    pJson.terminate()
    pool.join()