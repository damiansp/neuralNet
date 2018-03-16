import numpy as np
from keras.datasets import imdb
from keras.layers import Dense
from keras.models import Sequential

N_WORDS = 10000 # keep only the most frequent N_WORDS words
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=N_WORDS)

print('Sample datum:', X_train[0])
print('Sample label:', y_train[0])
print('Max word index:', max([max(seq) for seq in X_train])) # <= N_WORDS - 1

# Decode back to English
word_index = imdb.get_word_index()
reverse_word_index = {v: k for k, v in word_index.items()}
# Offset by 3 bc 0, 1, 2 reserved for 'padding', 'start', and 'unk'
print(' '.join([reverse_word_index.get(i - 3, '?') for i in X_train[0]]))


def vectorize_sequences(sequences, dimension=N_WORDS):
    results = np.zeros([len(sequences), dimension])
    for i, seq in enumerate(sequences):
        results[i, seq] = 1.
    return results

X_train = vectorize_sequences(X_train)
X_test  = vectorize_sequences(X_test)
y_train = np.asarray(y_train).astype('float32')
y_test  = np.asarray(y_test).astype('float32')

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(N_WORDS,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1,  activation='sigmoid'))
model.summary()
          
