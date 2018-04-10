import math
import os

import numpy as np
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


DATA_DIR = './data/'
N_TIMESTEPS = 20 # steps/data points per sample
HIDDEN_SIZE = 10
BATCH = 96 # = 24 hours
EPOCHS = 10

# NOTE: for stateful models, training and test set sizes must be exact multiples
# of the batch size


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
X_train, X_test, Y_train, Y_test = X[:split], X[split:], Y[:split], Y[split:]
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
less_mod.fit(X_train,
             Y_train,
             epochs=EPOCHS,
             batch_size=BATCH,
             validation_data=(X_test, Y_test),
             shuffle=False)
    

# Stateful fit
compile_mod(ful_mod)

# Training and Test set sizes must be multiples of BATCH
train_size = (X_train.shape[0] // BATCH) * BATCH
test_size  = (X_test.shape[0]  // BATCH) * BATCH
X_train, Y_train = X_train[:train_size], Y_train[:train_size]
X_test,  Y_test  = X_test[:test_size],   Y_test[:test_size]
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

for i in range(EPOCHS):
    print('Epoch: {:d}/{:d}'.format(i + 1, EPOCHS))
    ful_mod.fit(X_train,
                Y_train,
                batch_size=BATCH,
                epochs=1,
                validation_data=(X_test, Y_test),
                shuffle=False)

ful_score, _ = ful_mod.evaluate(X_test, Y_test, batch_size=BATCH)
rmse = math.sqrt(ful_score)
print('Stateful Model:\n  MSE: {:.3f}\n  RMSE: {:.3f}'.format(ful_score, rmse))
