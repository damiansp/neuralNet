import collections
import itertools
import os

import matplotlib.pyplot as plt
import nltk
import numpy as np
from keras.layers import Activation, Dense, Dropout, Input, Permute
from keras.layers.embeddings import Embedding
from keras.layers.merge import add, concatenate, dot
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils


DATA_DIR = './data/fb_bAbI/en/'
TRAIN_FILE = DATA_DIR + 'qa1_single-supporting-fact_train.txt'
TEST_FILE = DATA_DIR + 'qa1_single-supporting-fact_test.txt'
PAD = 0


def get_data(infile):
    stories, questions, answers = [], [], []
    story_text = []
    f_in = open(TRAIN_FILE, 'rb')
    for line in f_in:
        line = line.decode('utf-8').strip()
        line_number, text = line.split(' ', 1)
        if '\t' in text:
            question, answer, _ = text.split('\t')
            stories.append(story_text)
            questions.append(question)
            answers.append(answer)
            story_text = []
        else:
            story_text.append(text)
    f_in.close()
    return stories, questions, answers

data_train = get_data(TRAIN_FILE)
data_test  = get_data(TEST_FILE)


# Suspect the indentation may be wrong here
def build_vocab(train_data, test_data):
    counter = collections.Counter()
    for stories, questions, answers in [train_data, test_data]:
        for story in stories:
            for sentence in story:
                for word in nltk.word_tokenize(sentence):
                    counter[word.lower()] += 1
                for question in questions:
                    for word in nltk.word_tokenize(question):
                        counter[word.lower()] += 1
                for answer in answers:
                    for word in nltk.word_tokenize(answer):
                        counter[word.lower()] += 1
    word2idx = {w: (i + 1) for i, (w, _) in enumerate(counter.most_common())}
    word2idx['PAD'] = PAD
    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word

word2idx, idx2word = build_vocab(data_train, data_test)
vocab_size = len(word2idx)


def get_maxlens(train_data, test_data):
    story_maxlen, question_max_len = 0, 0
    for stories, questions, _ in [train_data, test_data]:
        for story in stories:
            story_len = 0
            for sentence in story:
                s_words = nltk.word_tokenize(sent)
                story_len += len(s_words)
            if story_len > story_maxlen:
                story_maxlen = story_len
        for question in questions:
            question_len = len(nltk.word_tokenize(question))
            if question_len > question_maxlen:
                question_maxlen = question_len
    return story_maxlen, question_maxlen

story_maxlen, question_max = get_maxlens(data_train, data_test)


def vectorize(data, word2idx, story_maxlen, question_maxlen):
    X_s, X_q, Y = [], [], []
    stories, questions, answers = data
    for story, question, answer in zip(stories, questions, answers):
        x_s = [[word2idx[w.lower()] for w in nltk.word_tokenize(s)]
               for s in story]
        x_s = list(itertools.chain.from_iterable(x_s))
        x_q = [word2idx[w.lower()] for w in nltk.word_tokenize(question)]
        X_s.append(x_s)
        X_q.append(x_q)
        Y.append(word2indx[answer.lower()])
    return (pad_sequence(X_s, maxlen=story_maxlen),
            pad_sequence(X_q, maxlen=question_maxlen),
            np_utils.to_categorical(Y, num_classes=len(word2idx)))

X_s_train, X_q_train, Y_train = vectorize(
    data_train, word2idx, story_maxlen, question_maxlen)
X_s_test,  X_q_test,  Y_train = vectorize(
    data_test,  word2idx, story_maxlen, question_maxlen)


