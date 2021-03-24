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

    
def train_step(X, y):
    with tf.GradientTape() as tape:
        logits = model(X, training=True) # preds
        loss = loss_object(y, logits)
    loss_history.append(loss.numpy().mean())
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


w = tf.Variable([[1.]])
with tf.GradientTape() as tape:
    loss = w * w                # d/dw[w^2] = 2w

tape.gradient(loss, w)          # tf.Tensor([[2.]], shape=(1, 1), dtype=float32)


x = tf.ones((2, 2))             # [[1 1]
with tf.GradientTape() as t:    #  [1 1]]
    t.watch(x)

y = tf.reduce_sum(x)            # 1 + 1 + 1 + 1 = 4
z = tf.square(y)                # 4^2 = 16
dzdx = t.gradient(z, x)         # dz/dx = [[8 8]
                                #          [8 8]]

x = tf.constant(3.)
with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y = x * x
    z = y * y

dzdx = t.gradient(z, x) # dz/dy dy/dx = 2y * 2x = 2x^2 * 2x = 4x^3 | @x=3: 108
# normally tape would be consumed in prev, but persists b/c persitent=True
dydx = t.gradient(y, x) # dy/dx = 2x | @x=3: 6
del t # drop ref to tape (bc persistent)


# higher order grads
x = tf.Variable(1.)
with tf.GradientTape() as t2:
    with tf.GradientTape() as t1:
        y = x ** 3                # y = x^3
    dydx = t1.gradient(y, x)      # dy/dx = 3x^2
d2ydx2 = t2.gradient(dydx, x)     # d2x/dx2 = x

