import matplotlib.pyplot as plt
import numpy as np
from keras import backend as k
from keras.datasets import mnist
from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils

EPOCHS = 20
BATCH = 128
VERBOSE = 1
OPTIMIZER = Adam()
VALIDATION_SPLIT = 0.2
IMG_HEIGHT, IMG_WIDTH = 28, 28
N_CHANNELS = 1
N_CLASSES = 10
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)


class LeNet:
    @staticmethod
    def build(input_shape, n_classes):
        model = Sequential()

        # CONV -> RELU -> POOL
        model.add(
            Conv2D(20, kernel_size=5, padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # CONV -> RELU -> POOL
        model.add(Conv2D(50, kernel_size=5, border_mode='same')) # <- padding?
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # FLAT -> DENSE -> RELU -> DENSE -> SOFTMAX
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))
        model.add(Dense(n_classes))
        model.add(Activation('softmax'))
        return model


(X_train, y_train), (X_test, y_test) = mnist.load_data()
k.set_image_dim_ordering('tf') # tf: [m, h, w, c]; th: [m, c, h, w]
X_train = X_train.astype('float32') / 255
X_test  = X_test.astype('float32')  / 255

# Input will be [60K, 28, 28, 1]
X_train = X_train[:, :, :, np.newaxis]
X_test  = X_test[:,  :, :, np.newaxis]
print('X_train:', X_train.shape)
print('X_test:',  X_test.shape)

y_train = np_utils.to_categorical(y_train, N_CLASSES)
y_test  = np_utils.to_categorical(y_test,  N_CLASSES)

model = LeNet.build(INPUT_SHAPE, N_CLASSES)
model.compile(
    loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
history = model.fit(X_train,
                    y_train,
                    batch_size=BATCH,
                    epochs=EPOCHS,
                    verbose=VERBOSE,
                    validation_split=VALIDATION_SPLIT)
score = model.evaluate(X_test, y_test, verbose=VERBOSE)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(history.history.keys())

plt.plot(history.history['acc'], 'k-', label='training accuracy')
plt.plot(history.history['val_acc'], 'r-', label='validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='best')
plt.show()

plt.plot(history.history['loss'], 'k-', label='training loss')
plt.plot(history.history['val_loss'], 'r-', label='validation loss')
plt.ylabel('Loss (categorical cross-entropy)')
plt.xlabel('Epoch')
plt.legend(loc='best')
plt.show()
