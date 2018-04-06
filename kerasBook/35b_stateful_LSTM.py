import math
import os

import numpy as np
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


DATA_DIR = './data'
N_TIMESTEPS = 20 # steps/data points per sample
HIDDEN_SIZE = 10
BATCH = 96 # = 24 hours
EPOCHS = 10


data = np.load(DATA_DIR + 'electricityConsumption.npy')
data = data.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
data = scaler.fit_transform(data)

X = np.zeros([data.shape[0], N_TIMESTEPS])
Y = np.zeros([data.shape[0], 1])
for i in range(len(data) - N_TIMESTEPS - 1):
    X[i] = data[i:i + N_TIMESTEPS].T
    Y[i] = data[i + N_TIMESTEPS + 1]

# Reshape X to 3D: [samples, timesteps, features]
X = np.expand_dims(X, axis=2)

# Train/Test
split = int(0.7 * len(data))
X_train, X_test, Y_train, Y_test = X[:sp], X[sp:], Y[:sp], Y[sp:]
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# Stateless Model----------------------------------------------------------
less_mod = Sequential()
less_mod.add(LSTM(
    HIDDEN_SIZE, input_shape=(N_TIMESTEPS, 1), return_sequences=False))
less_mod.add(Dense(1))


# Stateful Model-----------------------------------------------------------
ful_mod = Sequential()
ful_mod.add(LSTM(HIDDEN_SIZE,
                 stateful=True,
                 batch_input_shape=(BATCH, N_TIMESTEPS, 1),
                 return_sequences=False))
ful_mod.add(Dense(1))


def compile_mod(mod):
    mod.compile(loss='mean_squared_error',
                optimizer='adam',
                metrics=['mean_squared_error'])

    
# Stateless fit
compile_mod(less_mod)
les_mod.fit(X_train,
            Y_train,
            epochs=EPOCHS,
            batch_size=BATCH,
            validation_data=(X_test, Y_test),
            shuffle=False)
    

# Stateful fit
compile_mod(ful_mod)

