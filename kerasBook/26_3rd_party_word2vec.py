# Gensim installation:
# https://radimrehurek.com/gensim/install.html

# text8 corpus available at:
# http://mattmahoney.net/dc/text8.zip

import logging
import os
from gensim.models import KeyedVectors

DATA_DIR = '../data'
MAX_LEN = 50
EMBED_SIZE = 300
MIN_LEN = 30


class Text8Sentences:
    def __init__(self, fname, maxlen):
        self.fname = fname
        self.maxlen = maxlen

    def __iter__(self):
        with open(fname, 'rb') as ftext:
            text = ftext.read().split()
            sentences, words = [], []
            for word in text:
                # ... (code in book is buggy)


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
sentences = Text8Sentences(os.path.join(DATA_DIR, 'text8'), MAX_LEN)
model = word2vec.Word2Vec(sentences, size=EMBED_SIZE, min_count=MIN_COUNT)
model.init_sims(replace=True)
model.save('word2vec_gensim.bin')

# Now can load from memory
model = Word2Vec.load('word2vec_gensim.bin')

# See some words in model
print(model.vocab.keys()[:10])

# Get embedding for whatever word
print(model['woman'])

# Similar words
print(model.most_similar('woman'))

# Related
print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=10))

# Get cos similarities
print(model.similarity('girl', 'woman'))
print(model.similarity('girl', 'car'))
      
