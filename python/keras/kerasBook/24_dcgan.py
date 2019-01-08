import sys

import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.layers import (
    Activation, Convolution2D, BatchNormalization, Dense, Dropout, Flatten,
    Input, Reshape, UpSampling2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.regularizers import l1, l1_l2

from image_utils import (
    dim_ordering_fix, dim_ordering_input, dim_ordering_reshape,
    dim_ordering_unfix)

sys.path.append('./keras_adversarial')
from keras_adversarial import (
    AdversarialModel, AdversarialOptimizerAlternating,
    AdversarialOptimizerSimultaneous, gan_targets, image_grid_callback,
    normal_latent_sampling, simple_gan)



def gan_targets(n):
    '''
    Training targets: 
    [generator_fake, generator_real, discriminator_fake, discriminator_real]
    
    n: number of samples
    return: array of targets
    '''
    generator_fake = np.ones([n, 1])
    generator_real = np.zeros([n, 1])
    discriminator_fake = np.zeros([n, 1])
    discriminator_real = np.ones([n, 1])
    return [generator_fake, generator_real,
            discriminator_fake, discrinator_real]


def generator():
    c = 256
    g_input = Input(shape=[100])
    H = Dense(c * 14 * 14, kernel_initializer='glorot_normal')(g_input)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)
    H = dim_ordering_reshape(c, 14)(H)
    H = UpSampling2D(size=(2, 2))(H)
    H = Convolution2D(
        int(c / 2),
        kernel_size=3,
        strides=3,
        padding='same',
        init='glorot_uniform')(H)
    H = BatchNormalization(axis=1)(H)
    H = Activation('relu')(H)
    H = Convolution2D(
        1, kernel_size=1, strides=1, padding='same', init='glorot_uniform')(H)
    g_V = Activation('sigmoid')(H)
    return Model(g_input, g_V)


def discriminator(input_shape=(1, 28, 28), dropout=0.5):
    d_input = dim_ordering_input(input_shape, name='input_x')
    c = 512
    H = Convolution2D(
        int(c / 2),
        kernel_size=5,
        strides=5,
        padding='same',
        activation='relu')(d_input)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout)(H)
    H = Convolution2D(
        c,
        kernel_size=5,
        strides=5,
        padding='same',
        activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout)(H)
    H = Flatten()(H)
    H = Dense(int(c / 2))(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout)(H)
    d_V = Dense(1, activation='sigmoid')(H)
    return Model(d_input, d_V)


def process_mnist(x):
    x = x.astype(np.float32) / 255.
    return x


def mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return process_mnist(x_train), mnist_process(x_test)



if __name__ == '__main__':
    # z in R^100
    latent_dim = 100

    # x in R^{28 x 28}
    input_shape = (1, 28, 28)

    # generator: (z -> x)
    gen_mod = generator()

    # discriminator: (x -> y)
    disc_mod = discriminator(input_shape)

    # GAN (x -> y_fake, y_real); z idedally generated on GPU
    gan = simple_gan(gen_mod, disc_mod, normal_latent_sampling((latent_dim,)))
    gen_mod.summary()
    disc_mod.summary()
    gan.summary()
                     
    
    
