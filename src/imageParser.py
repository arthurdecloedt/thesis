import random

import requests
from lxml import html
import image_downloader
from multiprocessing import Pool, Value


def enqueue_tweets_proxy(url_queue, proxy_queue, trash_queue, img_queue):
    n = 0
    while (not url_queue.empty() and not proxy_queue.empty()):
        enqueue_image_url_proxy(url_queue.get(), proxy_queue, trash_queue)
        n += 1
        if n % 1000:
            print(n)


def enqueue_image_url_proxy(tweet_url, proxy_queue, trash_queue):
    proxy_tries = 20
    session = requests.Session()
    session.trust_env = False  # Don't read proxy settings from OS

    if proxy_queue.empty():
        raise Exception('no proxies available!')
    proxyIP = proxy_queue.get(block=True)
    proxy = {"http": "http://" + proxyIP, "https": "https://" + proxyIP}
    page = ""
    for a in range(proxy_tries + 1):
        try:

            page = session.get(tweet_url, proxies=proxy)
            break
        except Exception as e:
            print("tried:")
            print(proxy)
            print(e)
            if proxy_queue.empty():
                print('no more')
                raise Exception('no proxies available!')
            proxy = proxy_queue.get(block=True)
            proxy = {"http": "http://" + proxyIP, "https": "https://" + proxyIP}
            if a == proxy_tries:
                print("Aborting download after %s tries" % proxy_tries)
                return
            continue
    try:
        proxy_queue.put(proxy, block=True)
        doc = html.fromstring(page.content)
        img = doc.cssselect('meta[property="og:image"]')[0].get('content')

    except:
        print("error")
        return
    if 'profile' in img:
        return
    else:
        try:
            name = tweet_url.split("/")[-1]
            print(img)
            print(name)
            image_downloader.download_img_proxy(img, name, session, proxy)
        except Exception as e:
            print(str(e))


def enqueue_tweets(url_queue, img_queue):
    n = 0
    while not url_queue.empty():
        enqueue_image_url(url_queue.get())
        n += 1
        if n % 10 == 0:
            print(n)
            print(img_queue.qsize())


def enqueue_image_url(tweet_url,destination):
    page = requests.get(tweet_url)
    doc = html.fromstring(page.content)
    try:
        img = doc.cssselect('meta[property="og:image"]')[0].get('content')
    except Exception as e:
        # print(e)

        # print('url is %s ' %tweet_url)
        return

    if 'profile' in img:
        # with lock:
        #     counter.value -= 1
        return
    else:
        name = tweet_url.split("/")[-1]
        try:

            image_downloader.download_img(img, name,destination)
            return
        except Exception as e:
            print(e.args)
            image_downloader.download_img(img, name, destination)
