import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import boston_housing
from keras.layers import Dense
from keras.models import Sequential

K = 4 # k-folds
EPOCHS = 500
BATCH = 1

((train_data, train_targets),
 (test_data, test_targets)) = boston_housing.load_data()

print('Train:', train_data.shape)
print('   targets:', train_targets[:10])
print('Test:', test_data.shape)

# Normalize
mean = train_data.mean(axis=0)
sd = train_data.std(axis=0)

def normalize(x, mu, sigma):
    return (x - mu) / sigma

train_data = normalize(train_data, mean, sd)
test_data = normalize(test_data, mean, sd)

def build_model():
    mod = Sequential()
    mod.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    mod.add(Dense(64, activation='relu'))
    mod.add(Dense(1))

    mod.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return mod

# K-folds CV
n_valid = len(train_data) // K
all_mae_histories = []

for i in range(K):
    print('Fold:', i)
    val_data = train_data[i * n_valid : (i + 1) * n_valid]
    val_targets = train_targets[i * n_valid : (i + 1) * n_valid]
    partial_train_data = np.concatenate(
        [train_data[:i * n_valid], train_data[(i + 1) * n_valid:]], axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * n_valid], train_targets[(i + 1) * n_valid:]],
        axis=0)
    mod = build_model()
    history= mod.fit(partial_train_data,
                     partial_train_targets,
                     validation_data=(val_data, val_targets),
                     epochs=EPOCHS,
                     batch_size=BATCH,
                     verbose=1)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

average_mae_history = [np.mean([x[i] for x in all_mae_histories])
                       for i in range(EPOCHS)]
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for p in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous*factor + p*(1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# Final mod
mod = build_model()
mod.fit(train_data,
        train_targets,
        epochs=80,
        batch_size=16,
        verbose=1)
test_mse_score, test_mae_score = mod.evaluate(test_data, test_targets)
print(test_mae_score)
            
