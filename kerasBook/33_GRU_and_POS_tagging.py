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


s_wordfreqs, s_maxlen, s_records = parse_sentence(DATA_DIR
                                                  + 'treebank_sent.txt')
t_wordfreqs, t_maxlen, t_records = parse_sentence(DATA_DIR
                                                  + 'treebank_poss.txt')
print(len(s_wordfreqs), s_maxlen, s_records)
print(len(t_wordfreqs), t_maxlen, t_records)
