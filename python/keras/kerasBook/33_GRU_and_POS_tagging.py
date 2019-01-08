import collections
import os

import nltk
import numpy as np
from keras.layers import (
    Activation, Dense, Dropout, RepeatVector, SpatialDropout1D)
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

DATA_DIR = './data/'
MAX_SEQ_LEN = 250
S_MAX_FEATURES = 5000
T_MAX_FEATURES = 45
PAD = 0
UNK = 1
DROPOUT = 0.2
EMBED_SIZE = 128
HIDDEN_SIZE = 64
BATCH = 32
EPOCHS = 1


sent_data_file = open(DATA_DIR + 'treebank_sents.txt', 'wb')
pos_data_file  = open(DATA_DIR + 'treebank_poss.txt',  'wb')
sents = nltk.corpus.treebank.tagged_sent()
for sent in sents:
    words, poss = [], []
    for word, pos in sent:
        if pos == '-NONE-':
            continue
        words.append(word)
        poss.append(pos)
    sent_data_file.write('{:s}n'.format(' '.join(words)))
    pos_data_file.write('{:s}n'.format(' '.join(poss)))
sent_data_file.close()
pos_data_file.close()

# Explore data to find vocab sizes (for words and POSs)
def parse_sentences(filename):
    word_freqs = collection.Counter()
    n_records, maxlen = 0, 0
    in_file = open(filename, 'rb')
    for line in in_file:
        words = line.strip().lower().split()
        for word in words:
            word_freqs[word] += 1
        if len(words) > maxlen:
            maxlen = len(words)
        n_records += 1
    in_file.close()
    return word_freqs, maxlen, n_records


s_word_freqs, s_maxlen, s_records = parse_sentence(DATA_DIR
                                                  + 'treebank_sent.txt')
t_word_freqs, t_maxlen, t_records = parse_sentence(DATA_DIR
                                                  + 'treebank_poss.txt')
print(len(s_word_freqs), s_maxlen, s_records)
print(len(t_word_freqs), t_maxlen, t_records)

s_vocab_size = min(len(s_word_freqs), S_MAX_FEATURES) + 2 # +2 for UNK and PAD
s_word2index = {x[0]: i + 2
                for i, x in enumerate(s_word_freqs.most_common(S_MAX_FEATURES))}
s_word2index['PAD'] = PAD
s_word2index['UNK'] = UNK
s_index2word = {v: k for k, v in s_word2index.items()}

t_vocab_size = len(t_word_freqs) + 1
t_word2index = {x[0]: i
                for i, x in enumerate(t_word_freqs.most_common(T_MAX_FEATURES))}
t_word2index['PAD'] = PAD
t_index2word = {v: k for k, v in s_word2index.items()}


def build_tensor(
        filename, n_records, word2index, maxlen, make_categorical=False,
        n_classes=0):
    data = np.empty([n_records,], dtype=list)
    in_file = open(filename, 'rb')
    i = 0
    for line in in_file:
        wids = []
        for word in line.strip().lower().split():
            if word in word2index:
                wids.append(word2index[word])
            else:
                wids.append(word2index['UNK'])
        if make_categorical:
            data[i] = np_utils.to_categorical(wids, n_classes=n_classes)
        else:
            data[i] = wids
        i +=1
    in_file.close()
    padded_data = sequence.pad_sequences(data, maxlen=maxlen)
    return padded_data


X = build_tensor(
    DATA_DIR + 'treebank_sents.txt', s_records, s_word2index, MAX_SEQ_LEN)
Y = build_tensor(DATA_DIR + 'treebank_poss.txt',
                 t_records,
                 t_word2index,
                 True,
                 t_vocab_size)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1103)


# Build network model------------------------------------------------------
mod = Sequential()
mod.add(Embedding(s_vocab_size, EMBED_SIZE, input_length=MAX_SEQ_LEN))
mod.add(SpatialDropout1D(Dropout(DROPOUT)))
mod.add(GRU(HIDDEN_SIZE, dropout=DROPOUT, recurrent_dropout=DROPOUT))
mod.add(RepeatVector(MAX_SEQ_LEN))
mod.add(GRU(HIDDEN_SIZE, return_sequences=True))
mod.add(TimeDistributed(Dense(t_vocab_size)))
mod.add(Activation('softmax'))
mod.summary()

mod.compile(
    loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

mod.fit(X_train, Y_train, batch_size=BATCH)
score, acc = mod.evaluate(X_test, Y_test, batch_size=BATCH)
print('Test score: %.3f\tAccuracy: %.3f' % (score, acc))

        
