import requests


def download_imgs(imgQueue):
    a=1
    while(not imgQueue.empty()):
        a+=1
        imgurl= imgQueue.get()
        file = open("../out/imgs/%s.jpg" %a,'wb')
        file.write(requests.get(imgurl).content)
        file.close()


def download_img(imgurl,name):
    file = open("../out/imgs/%s.jpg" % name, 'wb')
    file.write(requests.get(imgurl).content)
    file.close()