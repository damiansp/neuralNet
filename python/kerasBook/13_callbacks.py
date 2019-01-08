from keras.callbacks import Callback, EarlyStopping
from keras.layers import Dense

EarlyStopping(
    monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

class LossHistory(Callback):
    def on_train_begin(self, logs{}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        model = Sequential()
        model.add(Dense(10, input_dim=784, init='glorot_uniform'))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', omptimzer='rmsprop')


history = LossHistory()
model.fit(X_train, Y_train, batch_size=128, n_epoch=20, verbose=0, mode='auto')
