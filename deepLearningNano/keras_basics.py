import numpy as np
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.utils import np_utils

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0],    [0],    [0],    [1]],    dtype=np.float32) # AND
y = np_utils.to_categorical(y)
print(y)

model = Sequential()
model.add(Dense(32, input_dim=X.shape[1]))
model.add(Activation('tanh'))
model.add(Dense(2))
model.add(Activation('sigmoid'))

model.compile(
    loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X, y, epochs=1000, verbose=0)
score = model.evaluate(X, y)
print('Accuracy:', score[-1])
print('Predictions:')
print(model.predict_proba(X))
