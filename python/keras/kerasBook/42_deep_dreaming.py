import os

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.applications import vgg16
from keras.layers import Input


DATA_DIR = './data/'
IMAGE_FILE = ['analytics.jpg', 'mnickerson.jpg', 'content.jpg', 'puppy.jpg'][0]
N_POOL_LAYERS = 5
ITERS_PER_LAYER = 5
STEP = 300


def preprocess(img):
    img4d = img.copy().astype('float64')
    if K.image_dim_ordering() == 'th': # for Theano backend
        # (H, W, C) -> (C, H, W)
        img4d = img4d.transpose((2, 0, 1))
    img4d = np.expand_dims(img4d, axis=0)
    img4d = vgg16.preprocess_input(img4d)
    return img4d


def deprocess(img4d):
    img = img4d.copy()
    img = img.reshape((img4d.shape[1], img4d.shape[2], img4d.shape[3]))
    if K.image_dim_ordering() == 'th':
        # (C, H, W)  -> (H, W, C)
        img = img.transpose((1, 2, 0))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    # BGR -> RGB
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img


img = plt.imread(DATA_DIR + IMAGE_FILE)
img_copy = img.copy()
p_img = preprocess(img_copy)
batch_shape = p_img.shape
print('Original image:', img.shape, '\nPreprocessed:', p_img.shape)

dream = Input(batch_shape=batch_shape)
mod = vgg16.VGG16(input_tensor=dream, weights='imagenet', include_top=False)
layer_dict = {layer.name: layer for layer in mod.layers}
print(layer_dict)

for n in range(N_POOL_LAYERS):
    # Identify pooling layer
    layer_name = 'block{:d}_pool'.format(n + 1)

    # Build loss function hat maximizes the mean activation in layer
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output)

    # Compute gradient of image w.r.t. loss and normalize
    grads = K.gradients(loss, dream)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # Def func to return loss and grad given input img
    f = K.function([dream], [loss, grads])
    img_value = p_img.copy()
    #fig, axes = plt.subplots(1, ITERS_PER_LAYER, figsize=(20, 10))
    for i in range(ITERS_PER_LAYER):
        loss_value, grads_value = f([img_value])
        img_value += grads_value * STEP
        #axes[i].imshow(deprocess(img_value))
        plt.imshow(deprocess(img_value))
        plt.show()


