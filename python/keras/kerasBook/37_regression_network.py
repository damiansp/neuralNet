# Predict atmospheric benzene levels from other atmospheric chemicals present
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, Input
from keras.models import Model
from sklearn.preprocessing import StandardScaler

DATA_DIR = './data/'
INFILE = 'AirQualityUCI.csv'
EPOCHS = 20
BATCH = 10

data = pd.read_csv(DATA_DIR + INFILE, sep=';', decimal=',', header=0)
data = data.drop(['Date', 'Time', 'Unnamed: 15', 'Unnamed: 16'], axis=1)
data = data.fillna(data.mean()) # NA <- col means
X_orig = data.as_matrix()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_orig)
X_means, X_sds = scaler.mean_, scaler.scale_

# Split x/y
y = X_scaled[:, 3]
X = np.delete(X_scaled, 3, axis=1) # same as X_scaled.drop('C6H6(GT)', axis=1)

train_size = int(0.7 * X.shape[0])
X_train, X_test, y_train, y_test = (
    X[:train_size], X[train_size:], y[:train_size], y[train_size:])


readings = Input(shape=(12,))
x = Dense(8, activation='relu', kernel_initializer='glorot_uniform')(readings)
benzene = Dense(1, kernel_initializer='glorot_uniform')(x)
mod = Model(inputs=[readings], outputs=[benzene])
print(mod.summary())
mod.compile(loss='mse', optimizer='adam')

history = mod.fit(
    X_train, y_train, batch_size=BATCH, epochs=EPOCHS, validation_split=0.2)

raw_pred = mod.predict(X_test).flatten()
for i in range(10):
    label = y_test[i]*X_sds[3] + X_means[3]
    prediction = raw_pred[i]*X_sds[3] + X_means[3]
    print('Benzene Concentration:\nObserved: {:.3f}, Predicted: {:.3f}'.format(
        label, prediction))

plt.plot(np.arange(y_test.shape[0]),
         y_test * X_sds[3] / X_means[3],
         color='k',
         label='actual')
plt.plot(np.arange(raw_pred.shape[0]),
         raw_pred * X_sds[3] / X_means[3],
         color='r',
         label='predicted')
plt.xlabel('Time')
plt.ylabel('Benzene concentration')
plt.legend()
plt.show()
          
