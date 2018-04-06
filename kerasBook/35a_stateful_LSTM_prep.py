# Stateful RNNs retain previous batch state as initial state for following batch
import re
import os

import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = './data/'
N_ENTRIES = 1000 # no. of data points in ts to use

data = []
cid = 250

data_file = open(DATA_DIR + 'electricityConsumption.txt', 'r')
for line in data_file:
    if line.startswith('"";"'):
        continue
    cols = [float(re.sub(',', '.', x)) for x in line.strip().split(';')[1:]]
    data.append(cols[cid])
data_file.close()

plt.plot(range(N_ENTRIES), data[:N_ENTRIES])
plt.ylabel('Electricity Consumption')
plt.xlabel('Time (x15min)')
plt.show()

np.save(DATA_DIR, 'electrictyConsumption.npy', np.array(data))
