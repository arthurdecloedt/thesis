import json_parser
import imageParser
import image_downloader
from queue import *
import multiprocessing

file_name = "data/aapl.json"
manager = multiprocessing.Manager()
url_queue = manager.Queue(60000)

proxy_queue = manager.Queue(60000)
trash_queue = manager.Queue(60000)
img_queue = manager.Queue(60000)

json_parser.get_tweet_queue(file_name,url_queue)
imageParser.enqueue_tweets(url_queue,img_queue)
image_downloader.download_imgs(img_queue)



