import numpy as np
from keras.layers import Activation, Dense, SimpleRNN
from keras.models import Sequential
from keras.utils.vis_utils import plot_model


SEQ_LEN = 10
STEP = 1

HIDDEN_SIZE = 128
BATCH = 128
ITERS = 25
EPOCHS_PER_ITER = 1
PREDS_PER_EPOCH = 100

f_in = open('data/alice_in_wonderland.txt', 'rb')
lines = []

for line in f_in:
    line = line.strip().lower()
    line = line.decode('ascii', 'ignore')
    if len(line) == 0:
        continue
    lines.append(line)
f_in.close()

print(lines[100:110])
text = ' '.join(lines)

chars = set([c for c in text])
n_chars = len(chars)
char2index = {c: i for i, c in enumerate(chars)}
index2char = {i: c for i, c in enumerate(chars)}

input_chars = []
label_chars = []

for i in range(0, len(text) - SEQ_LEN, STEP):
    input_chars.append(text[i:i + SEQ_LEN])
    label_chars.append(text[i + SEQ_LEN])

# gives a SEQ_LEN sequence of text and the character following, e.g.
# 'it turned into a pig':
# 'it turned ' -> 'i'
# 't turned i' -> 'n'...

X = np.zeros([len(input_chars), SEQ_LEN, n_chars], dtype=np.bool)
y = np.zeros([len(input_chars), n_chars], dtype=np.bool)

for i , input_char in enumerate(input_chars):
    for j, ch in enumerate(input_char):
        X[i, j, char2index[ch]] = 1
    y[i, char2index[label_chars[i]]] = 1

mod = Sequential()
mod.add(SimpleRNN(HIDDEN_SIZE,
                  return_sequences=False,
                  input_shape=(SEQ_LEN, n_chars),
                  unroll=True))
mod.add(Dense(n_chars))
mod.add(Activation('softmax'))

mod.compile(loss='categorical_crossentropy', optimizer='rmsprop')


for iteration in range(ITERS):
    print('=' * 50)
    print('Iteration: %d' % iteration)
    mod.fit(X, y, batch_size=BATCH, epochs=EPOCHS_PER_ITER)

    test_idx = np.random.randint(len(input_chars))
    test_chars = input_chars[test_idx]
    print('Generating from seed: %s' % test_chars)
    print(test_chars, end='')

    for i in range(PREDS_PER_EPOCH):
        X_test = np.zeros([1, SEQ_LEN, n_chars])
        for i, ch in enumerate(test_chars):
            X_test[0, i, char2index[ch]] = 1
        pred = mod.predict(X_test, verbose=0)[0]
        y_pred = index2char[np.argmax(pred)]
        print(y_pred, end='')

        # move forwad with test_chars + y_pred
        test_chars = test_chars[1:] + y_pred
print()
