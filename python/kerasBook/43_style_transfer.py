import os
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.applications import vgg16
from scipy.misc import imresize

DATA_DIR = './data/'
CONTENT_IMG = DATA_DIR + 'analytics.jpg'
STYLE_IMG = DATA_DIR + ['starrynight.jpg', 'gorilla.jpg', 'puppy.jpg'][0]
IMG_DIM = 400
N_LAYERS = 5
CONTENT_WEIGHT = 0.1
STYLE_WEIGHT = 5.
VAR_WEIGHT = 0.01
ITERATIONS = 5
ETA = 0.001

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


content_img = K.variable(preprocess(content_img_value))
style_img   = K.variable(preprocess(style_img_value))
if K.image_dim_ordering == 'th':
    comb_img = K.placeholder((1, 3, IMG_DIM, IMG_DIM))
else:
    comb_img = K.placeholder((1, IMG_DIM, IMG_DIM, 3))

# Concat images to a single input
input_tensor = K.concatenate([content_img, style_img, comb_img], axis=0)

mod = vgg16.VGG16(
    input_tensor=input_tensor, weights='imagenet', include_top=False)
layer_dict = {layer.name: layer.output for layer in mod.layers}


def content_loss(content, comb):
    return K.sum(K.square(comb - content)) # SSE


def gram_matrix(x):
    if K.image_dim_ordering() == 'th':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss_per_layer(style, comb):
    S = gram_matrix(style)
    C = gram_matrix(comb)
    channels = 3
    size = IMG_DIM * IMG_DIM
    #return K.sum(K.square(S - C)) / (4 * (channels**2) * (size**2))
    return content_loss(C, S) / (4 * (channels**2) * (size**2))


def style_loss():
    stl_loss = 0.
    for i in range(N_LAYERS):
        layer_name = 'block{:d}_conv1'.format(i + 1)
        layer_features = layer_dict[layer_name]
        style_features = layer_features[1, :, :, :]
        comb_features  = layer_features[2, :, :, :]
        stl_loss += style_loss_per_layer(style_features, comb_features)
    return stl_loss / N_LAYERS


def variation_loss(comb):
    if K.image_dim_ordering() == 'th':
        dx = K.square(comb[:, :, :IMG_DIM - 1, :IMG_DIM - 1]
                      - comb[:, :, 1:, :IMG_DIM - 1])
        dy = K.square(comb[:, :, :IMG_DIM - 1, :IMG_DIM -1]
                      - comb[:, :, IMG_DIM - 1, 1:])
    else:
       dx = K.square(comb[:, :IMG_DIM - 1, :IMG_DIM - 1, :]
                     - comb[:, 1:, :IMG_DIM - 1, :])
       dy = K.square(comb[:, :IMG_DIM - 1, :IMG_DIM - 1, :]
                     - comb[:, :IMG_DIM - 1, 1:, :])
    return K.sum(K.pow(dx + dy, 1.25))
                      
                      
c_loss = content_loss(content_img, comb_img)
s_loss = style_loss()
v_loss = variation_loss(comb_img)
loss = CONTENT_WEIGHT*c_loss + STYLE_WEIGHT*s_loss + VAR_WEIGHT*v_loss

grads = K.gradients(loss, comb_img)[0]
f = K.function([comb_img], [loss, grads])
content_img4d = preprocess(content_img_value)

for i in range(ITERATIONS):
    print('Epoch: {:d}/{:d}'.format(i + 1, ITERATIONS))
    loss_value, grads_value = f([content_img4d])
    content_img4d += grads_value * ETA
    plt.imshow(deprocess(content_img4d))
    plt.show()
