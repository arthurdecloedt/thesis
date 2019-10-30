import multiprocessing
from multiprocessing.pool import Pool

import imageParser
import image_downloader
import json_parser
from queue import *

file_name = "../data/aapl.json"


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
    counter = 1
    url = url_queue.get(block=True)
    pool.apply_async(imageParser.enqueue_image_url,(url,img_queue))

    while counter < 1000:
        url = url_queue.get(block=True)
        # a new url needs to be processed
        counter += 1
        pool.apply_async(imageParser.enqueue_image_url, (url, img_queue))
        print(counter)
    pool.close()
    pool.join()
while not img_queue.empty():
    print(img_queue.get())