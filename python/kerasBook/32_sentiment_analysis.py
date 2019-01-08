import collections
import os

import matplotlib.pyplot as plt
import nltk
import numpy as np
from keras.layers import Activation, Dense, Dropout, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split


DATA_DIR = './data/'
MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40
EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
DROPOUT = 0.2
BATCH = 32
EPOCHS = 10


# Explore data characteristics
maxlen = 0
word_freqs = collections.Counter()
n_recs = 0
f_train = open(DATA_DIR + 'umich-sentiment-train.txt', 'r')
for line in f_train:
    label, sentence = line.strip().split('\t')
    #words = nltk.word_tokenize(sentence.decode('ascii', 'ignore').lower())
    words = nltk.word_tokenize(sentence.lower())
    if len(words) > maxlen:
        maxlen = len(words)
    for word in words:
        word_freqs[word] += 1
    n_recs += 1
f_train.close()


vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word2index = {x[0]: i + 2
              for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index['PAD'] = 0
word2index['<UNK>'] = 1
index2word = {v: k for k, v in word2index.items()}

X = np.empty([n_recs, ], dtype=list)
y = np.zeros([n_recs, ])
i = 0
f_train = open(DATA_DIR + 'umich-sentiment-train.txt', 'r')
for line in f_train:
    label, sentence = line.strip().split('\t')
    #words = nltk.word_tokenize(sentence.decode('ascii', 'ignore').lower())
    words = nltk.word_tokenize(sentence.lower())
    seqs = []
    for word in words:
        seqs.append(word2index.get(word, word2index['<UNK>']))
        #if word2index.has_key(word):
        #    seqs.append(word2index[word])
        #else:
        #    seqs.append(word2index['<UNK>'])
    X[i] = seqs
    y[i] = int(label)
    i += 1
f_train.close()
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1103)


mod = Sequential()
mod.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))
mod.add(SpatialDropout1D(DROPOUT))
mod.add(LSTM(HIDDEN_LAYER_SIZE, dropout=DROPOUT, recurrent_dropout=DROPOUT))
mod.add(Dense(1))
mod.add(Activation('sigmoid'))
mod.summary()

mod.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = mod.fit(X_train,
                  y_train,
                  batch_size=BATCH,
                  epochs=EPOCHS,
                  validation_data=(X_test, y_test))

plt.subplot(212)
plt.plot(history.history['loss'], 'k-', label='train')
plt.plot(history.history['val_loss'], 'r-', label='valid')
plt.ylabel('Loss (binary cross-entropy)')
plt.legend()

plt.subplot(211)
plt.plot(history.history['acc'], 'k-', label='train')
plt.plot(history.history['val_acc'], 'r-', label='valid')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

score, acc = mod.evaluate(X_test, y_test, batch_size=BATCH)
print('Test score: %.3f\tAccuracy: %.3f' % (score, acc))

for i in range(5):
    idx = np.random.randint(len(X_test))
    xtest = X_test[idx].reshape(1, 40)
    label = y_test[idx]
    pred = mod.predict(xtest)[0][0]
    sent = ' '.join([index2word[x] for x in xtest[0].tolist() if x != 0])
    print('%.0f\t%d\t%s' % (pred, label, sent))
