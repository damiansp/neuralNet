# ...
from keras.layers.wrappers import Bidirectional
# ...

# ... (see 33)
mod = Sequential()
mod.add(Embedding(s_vocab_size, EMBED_SIZE, input_length=MAX_SEQ_LEN))
mod.add(SpatialDropout1D(Dropout(DROPOUT)))
mod.add(Bidirectional(LSTM(
    HIDDEN_SIZE, dropout=DROPOUT, recurrent_dropout=DROPOUT)))
mod.add(RepeatVector(MAX_SEQ_LEN))
mod.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True)))
mod.add(TimeDistributed(Dense(t_vocab_size)))
mod.add(Activation('softmax'))
