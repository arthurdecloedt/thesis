import io
import os

import requests
from PIL import Image


def download_imgs(imgQueue):
    a = 1
    while (not imgQueue.empty()):
        a += 1
        imgurl = imgQueue.get()
        file = open("../out/imgs/%s.jpg" % a, 'wb')
        file.write(requests.get(imgurl).content)
        file.close()


def download_img(imgurl, name):
    try:
        get = requests.get(imgurl)
        image_file = io.BytesIO(get.content)
        image = Image.open(image_file).convert('RGB')
        with open("../out/imgs/%s.jpg" % name, 'wb') as file:
            image.save(file,"JPEG", quality=85)
            file.close()
        print(name)
        print("size{0}".format(str(os.path.getsize("../out/imgs/%s.jpg" % name))))

    except Exception as e:
        print(e.with_traceback())





def download_img_proxy(imgurl, name,session,proxy):
    try:
        get = session.get(imgurl,proxies=proxy)
        image_file = io.BytesIO(get.content)
        image = Image.open(image_file).convert('RGB')
        with open("../out/imgs/%s.jpg" % name, 'wb') as file:
            image.save(file,"JPEG", quality=85)
            file.close()
        print(name)
        print("size{0}".format(str(os.path.getsize("../out/imgs/%s.jpg" % name))))

    except Exception as e:
        print(e.with_traceback())