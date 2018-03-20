import collections
import matplotlib.pyplot as plt
import nltk
import numpy as np
from keras.layers import (
    Conv1D, Dense, Dropout, GlobalMaxPooling1D, SpatialDropout1D)
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

Constants = collections.namedtuple(
    'Constants',
    'INPUT_FILE VOCAB_SIZE EMBED_SIZE N_FILTERS N_WORDS BATCH EPOCHS')
constants = Constants(
    INPUT_FILE='./data/umich-sentiment-train.txt',
    VOCAB_SIZE=5000,
    EMBED_SIZE=100,
    N_FILTERS=256,
    N_WORDS=3,
    BATCH=64,
    EPOCHS=20)

# Parse the data set
maxlen = 0
word_freqs = collections.Counter()
num_recs = 0
ftrain = open(constants.INPUT_FILE, 'r')
for line in ftrain:
    label, sentence = line.strip().split('\t')    
    words = nltk.word_tokenize(sentence.lower())
    if len(words) > maxlen:
        maxlen = len(words)
    for word in words:
        word_freqs[word] += 1
    num_recs += 1
ftrain.close()

'''
counter = collections.Counter()
fin = open(constants.INPUT_FILE, 'rb')
maxlen = 0
for line in fin:
    _, sent = line.strip().split('\t')
    words = [x.lower() for x in nltk.word_tokenize(sent)]
    if len(words) > maxlen:
        maxlen = len(words)
    for word in words:
        counter[word] += 1
fin.close()
'''

word2index = collections.defaultdict(int)
for wid, word in enumerate(word_freqs.most_common(constants.VOCAB_SIZE)):
    word2index[word[0]] = wid + 1
vocab_size = len(word2index) + 1
index2word = {v: k for k, v in word2index.items()}

xs, ys = [], []
fin = open(constants.INPUT_FILE, 'rb')
for line in fin:
    label, sent = line.strip().split('\t')
    ys.append(int(label))
    words = [x.lower() for x in nltk.word_tokens(sent)]
    wids = [word2index[word] for word in words]
    xs.append(wids)
fin.close()
X = pad_sequences(xs, maxlen=maxlen)
Y = np_utils.to_categorical(ys)

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


# Model
model = Sequential()
model.add(Embedding(vocab_size, EMBED_SIZE, input_length=maxlen))
model.add(SpatialDropout1D(Dropout(0.2)))
model.add(Conv1D(filters=constants.N_FILTERS,
                 kernel_size=constants.N_WORDS,
                 activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(2, activation='softmax'))

model.compile(
    loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train,
                    Y_train,
                    batch_size=constants.BATCH,
                    epochs=constants.EPOCHS,
                    validation_data=(X_test, Y_test))

