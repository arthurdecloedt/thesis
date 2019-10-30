import random

import requests
from lxml import html
import image_downloader




def enqueue_tweets_proxy(url_queue, proxy_queue,trash_queue,img_queue):
    n= 0
    while(not url_queue.empty() and not proxy_queue.empty()):
        enqueue_image_url_proxy(url_queue.get(),proxy_queue,trash_queue,img_queue)
        n+=1
        if n % 1000:
            print(n)

def enqueue_image_url_proxy(tweet_url,proxy_queue,trash_queue,img_queue):
    proxy_tries = 20

    if proxy_queue.empty():
        raise Exception('no proxies available!')
    proxy = proxy_queue.get()
    page=""
    for a in range(proxy_tries):
        try:
            page = requests.get(tweet_url,proxies=proxy)
            break
        except:
            trash_queue.put(proxy)
            if proxy_queue.empty():
                raise Exception('no proxies available!')
            proxy = proxy_queue.get()
            if a == proxy_tries:
                print("Aborting download after %s tries" % proxy_tries)
                return
            continue
    doc = html.fromstring(page.content)
    img = doc.cssselect('meta[property="og:image"]')[0].get('content')
    proxy_queue.put(proxy)
    if 'profile' in img:
        return
    else:
        img_queue.put()
        return


def enqueue_tweets(url_queue,img_queue):
    n= 0
    while(not url_queue.empty()):
        enqueue_image_url(url_queue.get(),img_queue)
        n+=1
        if n % 10 ==0:
            print(n)
            print(img_queue.qsize())

def enqueue_image_url(tweet_url,img_queue):
    page = requests.get(tweet_url)

    doc = html.fromstring(page.content)

    try :
        img = doc.cssselect('meta[property="og:image"]')[0].get('content')
    except:
        print("no meta tag")
        return
    if 'profile' in img:
        return
    else:
        image_downloader.download_img(img,str(random.randint(1000000,10000000)))

        return
