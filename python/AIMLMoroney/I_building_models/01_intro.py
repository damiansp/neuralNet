import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from   tensorflow.keras import Sequential
from   tensorflow.keras.layers import Dense

EPOCHS = 500

x = np.array([-1,  0, 1, 2, 3, 4], dtype=float)
y = np.array([-3, -1, 1, 3, 5, 7], dtype=float)
dense1 = Dense(1, input_shape=[1])
mod = Sequential(dense1)
mod.compile(optimizer='sgd', loss='mse')
print(mod.summary())
mod.fit(x, y, epochs=EPOCHS, verbose=0)
W, b = dense1.get_weights()
print(f'Weights:\n W: {W}\n b: {b}')
pred = mod.predict([10.])
plt.scatter(x, y)
plt.scatter(10., pred, color='r')
plt.show()
