model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(784, )))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_CLASSES))
model.add(Activation('softmax'))
model.summary()


          
