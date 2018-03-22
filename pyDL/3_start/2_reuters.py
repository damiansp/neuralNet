import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import reuters
from keras.layers import Dense
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

N_WORDS = 10000
N_CLASSES = 46
VALID_SIZE = 1000
EPOCHS = 20
BATCH = 512

# Training data is words as numeric representations; lables are 0-45 topic ids
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
    num_words=N_WORDS)
print(train_data[10])
word_index = reuters.get_word_index()
reverse_word_index = {v: k for k, v in word_index.items()}
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?')
                             for i in train_data[10]])
print(decoded_newswire)


def one_hot(sequences, dimension=N_WORDS):
    results = np.zeros([len(sequences), dimension])
    for i, seq in enumerate(sequences):
        results[i, seq] = 1.
    return results

X_train = one_hot(train_data)
X_test  = one_hot(test_data)
Y_train = one_hot(train_labels, dimension=N_CLASSES)
Y_test  = one_hot(test_labels,  dimension=N_CLASSES)

# Same as:
#X_train = to_categorical(train_data)
#X_test  = to_categorical(test_data)
#Y_train = to_categorical(train_labels)
#Y_test  = to_categorical(test_labels)


X_valid = X_train[:VALID_SIZE]
X_train = X_train[VALID_SIZE:]
Y_valid = Y_train[:VALID_SIZE]
Y_train = Y_train[VALID_SIZE:]

mod = Sequential()
mod.add(Dense(64, activation='relu', input_shape=(N_WORDS,)))
mod.add(Dense(64, activation='relu'))
mod.add(Dense(N_CLASSES, activation='softmax'))

mod.compile(
    optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

history = mod.fit(X_train,
                  Y_train,
                  epochs=EPOCHS,
                  batch_size=BATCH,
                  validation_data=(X_valid, Y_valid))

plt.plot(history.history['loss'], 'k-', label='Train')
plt.plot(history.history['val_loss'], 'r-', label='Valid')
plt.ylabel('Categorical Cross-Entropy')
plt.legend()
plt.show()

preds = mod.predict(X_test)
