from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential

mod = Sequential()
mod.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
mod.add(MaxPooling2D((2, 2)))
mod.add(Conv2D(64, (3, 3), activation='relu'))
mod.add(MaxPooling2D((2, 2)))
mod.add(Conv2D(64, (3, 3), activation='relu'))
mod.add(Flatten())

mod.add(Dense(64, activation='relu'))
mod.add(Dense(10, activation='softmax'))

mod.summary()


test_loss, test_acc = mod.evaluate(test_images, test_labels)
print(test_acc)
