import numpy as np
import tensorflow as tf


# Training data
X_train = np.array([-1.,  0., 1., 2., 3., 4.], dtype=float)
y_train = np.array([-3., -1., 1., 3., 5., 7.], dtype=float)

# Trainable params
w = tf.Variable(np.random.random(), trainable=True)
b = tf.Variable(np.random.random(), trainable=True)

# Loss func
def abs_err(target, pred):
    return tf.abs(target - pred)

ETA = 0.001


# Grad desc.
def fit_data(X, y):
    with tf.GradientTape(persistent=True) as tape:
        pred = w*X + b
        loss = abs_err(y, pred)
    w_grad = tape.gradient(loss, w)
    b_grad = tape.gradient(loss, b)
    w.assign_sub(w_grad * ETA) # w -= ...
    b.assign_sub(b_grad * ETA)
    
    
for _ in range(500):
    fit_data(X_train, y_train)
    print(f'Est: y = {w.numpy()} + {b.numpy()}')

    
