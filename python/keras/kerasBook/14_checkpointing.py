# Allow saving at various points in case of failure

import numpy as np
import os
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils

BATCH = 128
EPOCHS = 20
MODEL_DIR = './tmp'

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784).astype('float32') / 255
X_test  = X_test.reshape(10000,  784).astype('float32') / 255
Y_train = np_utils.to_categorical(y_train, 10)
Y_test  = np_utils.to_categorical(y_test,  10)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

model = Sequential()
model.add(Dense(512, input_shape=(784,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save best model
checkpoint = ModelCheckpoint(filepath=os.path.join(MODEL_DIR,
                                                   'model-{epoch:02d}.h5'),
                             save_best_only=True)
model.fit(X_train,
          Y_train,
          batch_size=BATCH,
          nb_epoch=EPOCHS,
          validation_split=0.1,
          callbacks=[checkpoint])
          
