import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils

class LeNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()

        # CONV -> RELU -> POOL
        model.add(
            Conv2D(20, kernel_size=5, padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
