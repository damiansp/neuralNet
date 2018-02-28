import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.layers import (
    Activation, Conv2D,Dense, Dropout, Flatten, MaxPooling2D)
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import np_utils

# CIFAR-10 data set: 60K 32x32x3 images
CHANNELS = 3
HEIGHT, WIDTH = 32, 32
BATCH = 128
EPOCHS = 20
N_CLASSES = 10
VERBOSE = 1
VALID_SPLIT = 0.2
OPT = RMSprop()

# Load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train:', X_train.shape)
print('X_test:',  X_test.shape)

# Preprocess
# y -> one-hot
Y_train = np_utils.to_categorical(y_train, N_CLASSES)
Y_test  = np_utils.to_categorical(y_test,  N_CLASSES)

# Normalize X
X_train = X_train.astype('float32') / 255
X_test  = X_test.astype('float32')  / 255


# Model
model = Sequential()
model.add(
    Conv2D(32, (3, 3), padding='same', input_shape=(HEIGHT, WIDTH, CHANNELS)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(N_CLASSES))
model.add(Activation('softmax'))
model.summary()

# Train
model.compile(
    loss='categorical_crossentropy', optimizer=OPT, metrics=['accuracy'])
history = model.fit(X_train,
                    Y_train,
                    batch_size=BATCH,
                    epochs=EPOCHS,
                    validation_split=VALID_SPLIT,
                    verbose=VERBOSE)
score = model.evaluate(X_test, Y_test, batch_size=BATCH, verbose=VERBOSE)
print('Test score: %.4f; accuracy: %.4f' % (score[0], score[1]))

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

# Save model
#model_json = model.to_json()
#open('cifar10_architecture.json', 'w').write(model_json)
#model.save_weights('cifar10_weight.h5, overwrite=True)
