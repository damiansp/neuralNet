import collections
import os

import matplotlib.pyplot as plt
import nltk
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Inuput, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layes.wrappers import Bidirectional
from keras.models import Model
from keras.preprocessing import sequence
from scipy.stats import describe
from sklearn.model_selection import train_test_split

