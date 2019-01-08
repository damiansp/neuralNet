import numpy as np
from keras.datasets import mnist
from keras.layers.core import Activation, Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils

#np.random.seed(1671)

# Network and training
EPOCHS = 200
BATCH_SIZE = 128
VERBOSE = 1
N_CLASSES = 10
OPTIMIZER = SGD()
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2

# Data: suffle and split train/test
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train 60000 records of 28x28 images -> 60000 x 784
RESHAPED = 28 * 28
X_train = X_train.reshape(60000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.reshape(10000, RESHAPED)
X_test = X_test.astype('float32')

# Normalize
N_COLORS = 255
X_train /= N_COLORS
X_test /= N_COLORS

print('X_train:', X_train.shape)
print('X_test:', X_test.shape)

# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, N_CLASSES)
Y_test = np_utils.to_categorical(y_test, N_CLASSES)

# Softmax to 10 outputs
model = Sequential()
model.add(Dense(N_CLASSES, input_shape=(RESHAPED,)))
model.add(Activation('softmax'))
model.summary()

# Loss functions:
# https://keras.io/losses/

model.compile(
    loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

history = model.fit(X_train,
                    Y_train,
                    batch_size=BATCH_SIZE,
                    verbose=VERBOSE,
                    validation_split=VALIDATION_SPLIT)

score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print('Test score:', score[0])
print('Test accuracy:', score[1])
