




import json
from queue import *
import dateutil


def get_tweet_queue(jsonfile, tweetmap):
    with open(jsonfile, "r", encoding="utf-8") as file:
        n = 0
        while True:
            line = file.readline()
            # for a in range(14):
            #     line= line +  file.readline().strip('\n').strip().replace(' ','')
            # print(line)
            if not line:
                break
            # if n<450000:
            #     n+=1
            #     continue

            tweetmap[json.loads(line)['id']] = (
            json.loads(line)['content'], dateutil.parser.parse(json.loads(line)['date']['$date'], ignoretz=True))


tweetmap = {}
get_tweet_queue('/content/drive/My Drive/thesis/data/aapl.json', tweetmap)

# %%

import json
from queue import *
from sortedcontainers import SortedListWithKey

# %%

len(tweetmap)

# %%

!cat / content / drive / My\ Drive / thesis / data / aapl.json | head - 3

# %%

from os import listdir
from os.path import isfile, join

mypath = '/dev/shm/aapl/outscraper'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

analyser = SentimentIntensityAnalyzer()

import PIL
import tensorflow as tf
import numpy as np
import os
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, Input, Conv2D
from tensorflow.python.keras.applications import MobileNet

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop
import tensorflow.python.keras.backend as K

K.clear_session()

# %%

from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage import io
from skimage.transform import resize
from keras.preprocessing.image import DirectoryIterator


class ImageWithNames(DirectoryIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filenames_np = np.array(self.filepaths)
        self.class_mode = None  # so that we only get the images back
        self.batch_size = 65

    def _get_batches_of_transformed_samples(self, index_array):
        return (super()._get_batches_of_transformed_samples(index_array),
                self.filenames_np[index_array])


def get_Gen(tweetmap):
    analyser = SentimentIntensityAnalyzer()

    imagegen = ImageDataGenerator(rescale=1. / 255)
    datagen = ImageWithNames('/dev/shm/aapl/', imagegen, target_size=(224, 224))

    while True:
        img, name = datagen.next()
        sent_v = np.zeros(len(img))
        i = 0
        for n in name:
            imgid = n.split('/')[-1].split('.')[0]
            content = tweetmap[imgid][0]
            sent = analyser.polarity_scores(content)['compound']
            sentarr = [ sent['neg'] ,sent['neu'] ,sent['pos'] ,sent['compound']]

            i += 1
        yield [img], [sent_v]


# %%

def get_gen_autogen():
    imagegen = ImageDataGenerator(rescale=1. / 255)
    datagen = imagegen.flow_from_directory('/dev/shm/aapl/', target_size=(224, 224), class_mode=None, batch_size=32)
    while True:
        try:
            img = datagen.next()
        # embedded = mobilenet.predict(img)
        except Exception as e:
            print('failed to collect image,skipping batch')
            continue
        yield [img], [img]


# %%

from tensorflow.keras.layers import MaxPool2D, UpSampling2D
from tensorflow.keras.losses import binary_crossentropy

img_in = Input((224, 224, 3))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(img_in)
x = MaxPool2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPool2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPool2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPool2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(3, (1, 1), activation='relu', padding='same')(x)
autoenc = Model(img_in, x)
autoenc.summary()

# %%

# import time

# for a in range(1000):
#   print('tf')
#   time.sleep(0.01)
# import sys
# sys.stdout.flush()
autogen = get_gen_autogen()
autoenc.compile(optimizer='adam', loss='binary_crossentropy')
autoenc.fit_generator(autogen, steps_per_epoch=25, epochs=100,
                      callbacks=[tf.keras.callbacks.TensorBoard(log_dir='./logs')])

# %%

arr = next(get_gen_autogen())[0][0]
a = autoenc.predict(arr)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(arr[i].reshape(224, 224, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(a[i].reshape(224, 224, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# %%

from sortedcontainers import SortedKeyList

folder = '/dev/shm/aapl/outscraper'
folder1 = '/dev/shm/aapl/outscraperAAPL1'
filenames = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
filenames.extend([f for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))])
namelist = SortedKeyList(key=lambda x: x[0])
a = []
for f in filenames:
    imgid = f.split('/')[-1].split('.')[0]
    content, date = tweetmap[imgid]
    namelist.add((date, imgid, content))
    a.append(date.year)
print(len(a))

# %%

plt.hist(a, density=True, facecolor='g')
plt.show()

# %%

for l in model_str.layers:
    l.trainable = False
model_str.layers[-1].trainable = True

# %%

model_c = model_str.compile(loss='MSE', metrics=['MSE', 'MAE'])

# %%

from tensorflow.keras.callbacks import *

from keras import backend as K

valgen = get_Gen(tweetmap)
x, y = next(valgen)

model_str.fit_generator(valgen, steps_per_epoch=100, epochs=50, shuffle=True,
                        callbacks=[tf.keras.callbacks.TensorBoard(log_dir='./logs')])

# %%

% tensorboard - -logdir. / logs


# %%

!rm - r. / logs

# %%


