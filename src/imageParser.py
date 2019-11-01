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

def enqueue_image_url_proxy(tweet_url,proxy_queue,trash_queue):
    proxy_tries = 20
    session = requests.Session()
    session.trust_env = False  # Don't read proxy settings from OS

    if proxy_queue.empty():
        raise Exception('no proxies available!')
    proxy = proxy_queue.get(block=True)
    page=""
    for a in range(proxy_tries+1):
        try:
            page = session.get(tweet_url,proxies={"http": "http://" + proxy, "https": "https://" + proxy})
            break
        except:
            return
            print('tried')
            if proxy_queue.empty():
                raise Exception('no proxies available!')
            proxy = proxy_queue.get(block=True)
            if a == proxy_tries:
                print("Aborting download after %s tries" % proxy_tries)
                return
            continue
    try:
        print('got it')
        proxy_queue.put(proxy,block=True)
        doc = html.fromstring(page.content)
        img = doc.cssselect('meta[property="og:image"]')[0].get('content')

    except:
        print("error")
        return
    if 'profile' in img:
        return
    else:

        print("yasss")
        name = tweet_url.split("/")[-1]
        image_downloader.download_img(img, name,session,proxy)


def enqueue_tweets(url_queue,img_queue):
    n= 0
    while(not url_queue.empty()):
        enqueue_image_url(url_queue.get(),img_queue)
        n+=1
        if n % 10 ==0:
            print(n)
            print(img_queue.qsize())

def enqueue_image_url(tweet_url):
    page = requests.get(tweet_url)

    doc = html.fromstring(page.content)

    try :
        img = doc.cssselect('meta[property="og:image"]')[0].get('content')
    except:
        # with lock:
        #     counter.value -= 1
        return

        return
    if 'profile' in img:
        # with lock:
        #     counter.value -= 1
        return
    else:
        name = tweet_url.split("/")[-1]
        image_downloader.download_img(img,name)
        # with lock:
        #     counter.value -= 1
        # print("counter val %s" % counter.value)
        return
