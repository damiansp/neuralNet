GLOVE_MODEL = '../data/glove.6B.300d.txt'

glove_file = open(GLOVE_MODEL, 'rb')
word2embedding = {}
for line in glove_file:
    cols = line.strip().split()
    word = cols[0]
    embedding = np.array(cols[1:], dtype='float32')
    word2embedding[word] = embedding
glove_file.close()

embedding_weights = np.zeros([VOCAB_SIZE, EMBED_SIZE])
for word, index in word2index.items():
    try:
        embedding_weights[index, :] = word2emb[word]
    except KeyError:
        pass
