from keras.layers import Dense, Dot, Merge, Reshape
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.text import *
from keras.preprocessing.sequence import skipgrams

VOCAB_SIZE = 5000
EMBED_SIZE = 300

word_model = Sequential()
word_model.add(Embedding(VOCAB_SIZE,
                         EMBED_SIZE,
                         embeddings_initializer='glorot_uniform',
                         input_length=1))
word_model.add(Reshape((EMBED_SIZE,)))

context_model = Sequential()
context_model.add(Embedding(VOCAB_SIZE,
                            EMBED_SIZE,
                            embeddings_initializer='glorot_uniform',
                            input_length=1))
context_model.add(Reshape((EMBED_SIZE,)))

model = Sequential()
model.add(Merge([word_model, context_model], mode='dot'))
model.add(Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam')

text = 'I love green eggs and ham .'
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

# Tokenizer crates mapping of individual words to int IDs available in
# word_index attribute
word2id = tokenizer.word_index
id2word = {v: k for k, v in word2id.items()}
wids = [word2id[w] for w in text_to_word_sequence(text)]
pairs, labels = skipgrams(wids, len(word2id))
print(len(pairs), len(labels))

for i in range(10):
    print('({:s} ({:d}), {:s} ({:d})) -> {:d}'.format(
        id2word[pairs[i][0]], pairs[i][0],
        id2word[pairs[i][1]], pairs[i][1],
        labels[i]))
    
