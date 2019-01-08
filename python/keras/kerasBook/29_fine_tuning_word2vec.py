import collections
import matplotlib.pyplot as plt
import nltk
import numpy as np
from gensim.models import KeyedVectors
from keras.layers import (
    Conv1D, Dense, Dropout, GlobalMaxPooling, SpatialDropout)
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

INPUT_FILE = './data/umich-sentiment-train.txt'
WORD2VEC_MOD = './data/GoogleNews-vectors-negative300.bin.gz'
VOCAB_SIZE = 5000
EMBED_SIZE = 300
N_FILTERS = 256
N_WORDS = 3
BATCH = 64
EPOCHS = 10

counter = collections.Counter()

f_in = open(INPUT_FILE, 'rb')
max_len = 0
for line in f_in:
    _, sent = line.strip().split('\t')
    words = [x.lower() for x in nltk.word_tokenize(sent)]
    if len(words) > max_len:
        max_len = len(words)
    for word in words:
        counter[word] += 1
f_in.close()

word2index = collections.defaultdict(int)
for wid, word in enumerate(counter.most_common(VOCAB_SIZE)):
    word2index[word[0]] = wid + 1
vocab_size = len(word2index) + 1
index2word = {v: k for k, v in word2index.items()}

xs, ys = [], []

# Merge with above
f_in = open(INPUT_FILE, 'rb')
for line in f_in:
    label, sent = line.strip().split('\t')
    ys.append(int(label))
    words = [x.lower() for x in nltk.word_tokenize(sent)]
    wids = [word2index[word] for word in words]
    xs.append(wids)
f_in.close()

X = pad_sequences(xs, maxlen=max_len)
Y = np_utils.to_categorical(ys)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


# Load word2vec mod
word2vec = Word2Vec.load_word2vec_format(WORD2VEC_MOD, binary=True)
embedding_weights = np.zeros([vocab_size, EMBED_SIZE])
for word, index in word2index.items():
    try:
        embedding_weights[index, :] = word2vec[word]
    except KeyError:
        pass


mod = Sequential()
mod.add(Embedding(
    vocab_size, EMBED_SIZE, input_len=max_len, weights=[embedding_weights]))
mod.add(SpatialDropout1D(Dropout(0.2)))
mod.add(Conv1D(filters=N_FILTERS, kernel_size=N_WORDS, activation='relu'))
mod.add(GlobalMaxPooling1D())
mod.add(Dense(2, activation='softmax'))
mod.summary()

mod.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = mod.fit(X_train,
                  Y_train,
                  batch_size=BATCH,
                  epochs=EPOCHS,
                  validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=1)
print('Test score: {:.3f}, accuracy: {:.3f}'.format(score[0], score[1]))
        
