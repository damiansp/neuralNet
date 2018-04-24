import os
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.applications import vgg16
from scipy.misc import imresize

DATA_DIR = './data/'
CONTENT_IMG = DATA_DIR + 'analytics.jpg'
STYLE_IMG = DATA_DIR + 'puppy.jpg'
IMG_DIM = 400

content_img_value = imresize(plt.imread(CONTENT_IMG), (IMG_DIM, IMG_DIM))
style_img_value   = imresize(plt.imread(STYLE_IMG),   (IMG_DIM, IMG_DIM))
plt.subplot(121)
plt.imshow(content_img_value)
plt.subplot(122)
plt.imshow(style_img_value)
plt.show()


def preprocess(img):
    img4d = img.copy()
    img4d = img4d.astype('float64')
    if K.image_dim_ordering() == 'th':
        # (H, W, C) -> (C, H, W)
        img4d = img4d.transpose((2, 0, 1))
    img4d = np.expand_dims(img4d, axis=0)
    img4d = vgg16.preprocess_input(img4d)
    return img4d


def deprocess(img4d):
    img = img4d.copy()
    img = img.reshape((img4d.shape[1], img4d.shape[2], img4d.shape[3]))
    if K.image_dim_ordering() == 'th':
        # (C, H, W) -> (H, W, C)
        img = img.transpose((1, 2, 0))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    # BGR -> RGB
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img
