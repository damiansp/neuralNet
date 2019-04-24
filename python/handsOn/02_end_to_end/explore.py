import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HOME = os.environ['HOME']
DATA = f'{HOME}/Learning/neuralNet/data/kaggle/titanic'

df = pd.read_csv(f'{DATA}/train.csv')

fig = plt.figure(figsize=(18, 6), dpi=1600)
alpha = alpha_scatterplot = 0.2
alpha_barchart = 0.55
fig = plt.figure()
ax = fig.add_subplot(111)

