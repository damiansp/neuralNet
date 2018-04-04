import matplotlib.pyplot as plt
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator

DATA_DIR = '../data/kaggle_catdog_small/'
TRAIN_DIR = DATA_DIR + 'train/'
VALID_DIR = DATA_DIR + 'valid/'
IMAGE_DIMS = [150, 150, 3]
DROPOUT = 0.5
ETA = 1e-4
BATCH = 16
EPOCHS = 100
STEPS_PER_EPOCH = 100

mod = Sequential()
mod.add(Conv2D(32, (3, 3), activation='relu', input_shape=IMAGE_DIMS))
mod.add(MaxPooling2D((2, 2)))
mod.add(Conv2D(64, (3, 3), activation='relu'))
mod.add(MaxPooling2D((2, 2)))
mod.add(Conv2D(128, (3, 3), activation='relu'))
mod.add(MaxPooling2D((2, 2)))
mod.add(Conv2D(128, (3, 3), activation='relu'))
mod.add(MaxPooling2D((2, 2)))
mod.add(Flatten())
mod.add(Dropout(DROPOUT))
mod.add(Dense(512, activation='relu'))
mod.add(Dense(1, activation='sigmoid'))
mod.summary()

mod.compile(
    loss='binary_crossentropy', optimizer=RMSprop(lr=ETA), metrics=['acc'])


# Data processing/management
# Rescaling colors and data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                    target_size=IMAGE_DIMS[:2],
                                                    batch_size=BATCH,
                                                    class_mode='binary')
validation_generator = train_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMAGE_DIMS[:2],
    batch_size=BATCH,
    class_mode='binary')
history = mod.fit_generator(train_generator,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            epochs=EPOCHS,
                            validation_data=validation_generator,
                            validation_steps=STEPS_PER_EPOCH / 2)
mod.save('cats_and_dogs_small_1.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(121)
plt.plot(range(EPOCHS), acc, 'k-', label='Training')
plt.plot(range(EPOCHS), val_acc, 'r-', label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross-Entropy')
plt.legend()

plt.subplot(122)
plt.plot(range(EPOCHS), loss, 'k-', label='Training')
plt.plot(range(EPOCHS), val_loss, 'r-', label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

