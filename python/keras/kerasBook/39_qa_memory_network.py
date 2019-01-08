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
EMBEDDING_SIZE = 64
LATENT_SIZE = 32
DROPOUT = 0.3
BATCH = 32
EPOCHS = 50
N_DISPLAY = 10


def get_data(infile):
    stories, questions, answers = [], [], []
    story_text = []
    f_in = open(infile, 'rb')
    for line in f_in:
        line = line.decode('utf-8').strip()
        _, text = line.split(' ', 1)
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
    word2idx = {w: i + 1 for i, (w, _) in enumerate(counter.most_common())}
    word2idx['PAD'] = PAD
    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word

word2idx, idx2word = build_vocab(data_train, data_test)
vocab_size = len(word2idx)


def get_maxlens(train_data, test_data):
    story_maxlen, question_maxlen = 0, 0
    for stories, questions, _ in [train_data, test_data]:
        for story in stories:
            story_len = 0
            for sentence in story:
                s_words = nltk.word_tokenize(sentence)
                story_len += len(s_words)
            if story_len > story_maxlen:
                story_maxlen = story_len
        for question in questions:
            question_len = len(nltk.word_tokenize(question))
            if question_len > question_maxlen:
                question_maxlen = question_len
    return story_maxlen, question_maxlen

story_maxlen, question_maxlen = get_maxlens(data_train, data_test)


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
        Y.append(word2idx[answer.lower()])
    return (pad_sequences(X_s, maxlen=story_maxlen),
            pad_sequences(X_q, maxlen=question_maxlen),
            np_utils.to_categorical(Y, num_classes=len(word2idx)))

X_s_train, X_q_train, Y_train = vectorize(
    data_train, word2idx, story_maxlen, question_maxlen)
X_s_test,  X_q_test,  Y_test = vectorize(
    data_test,  word2idx, story_maxlen, question_maxlen)



# Model--------------------------------------------------------------------
# Inputs
story_input = Input(shape=(story_maxlen,))
question_input = Input(shape=(question_maxlen,))

# Story encoder memory
story_encoder = Embedding(input_dim=vocab_size,
                          output_dim=EMBEDDING_SIZE,
                          input_length=story_maxlen)(story_input)
story_encoder = Dropout(DROPOUT)(story_encoder)

# Question encoder
question_encoder = Embedding(input_dim=vocab_size,
                             output_dim=EMBEDDING_SIZE,
                             input_length=question_maxlen)(question_input)
question_encoder = Dropout(DROPOUT)(question_encoder)

# Match between story and question
match = dot([story_encoder, question_encoder], axes=[2, 2])

# Encode story into vector space of question
story_encoder_c = Embedding(input_dim=vocab_size,
                            output_dim=question_maxlen,
                            input_length=story_maxlen)(story_input)
story_encoder_c = Dropout(DROPOUT)(story_encoder_c)

# Combine match and story vectors
response = add([match, story_encoder_c])
response = Permute((2, 1))(response)

# Combine response and question vectors
answer = concatenate([response, question_encoder], axis=-1)
answer = LSTM(LATENT_SIZE)(answer)
answer = Dropout(DROPOUT)(answer)
answer = Dense(vocab_size)(answer)
output = Activation('softmax')(answer)

mod = Model(inputs=[story_input, question_input], outputs=output)
mod.compile(
    optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


history = mod.fit([X_s_train, X_q_train],
                  [Y_train],
                  batch_size=BATCH,
                  epochs=EPOCHS,
                  validation_data=([X_s_test, X_q_test], [Y_test]))

y_test = np.argmax(Y_test, axis=1)
preds = mod.predict([X_s_test, X_q_test])
preds = np.argmax(preds, axis=1)


for i in range(N_DISPLAY):
    story = ' '.join([idx2word[x] for x in X_s_test[i].tolist() if x != 0])
    question = ' '.join([idx2word[x] for x in X_q_test[i].tolist()])
    label = idx2word[y_test[i]]
    prediction = idx2word[preds[i]]
    print(story, question, label, prediction)
