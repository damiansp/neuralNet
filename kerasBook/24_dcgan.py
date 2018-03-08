import matplotlib as mpl
import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.layers import (
    Activation, Convolution2D, BatchNormalization, Dense, Dropout, Flatten,
    Input, LeakyReLU, Reshape, UpSampling2D)
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.regulaizers import l1, l1l2

mpl.use('Agg')
