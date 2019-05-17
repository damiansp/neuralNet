import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

ETA = 0.6
EPOCHS = 16

n = 1000
xs = []
ys = []
for i in range(n):
    W = 0.1
    b = 0.4
    x1 = np.random.normal(0., 1.)
    noise = np.random.normal(0., 0.05)
    y1 = W*x1 + b + noise
    xs.append(x1)
    ys.append(y1)
plt.plot(xs, ys, 'bo', alpha=0.2)
plt.show()


W = tf.Variable(tf.random_uniform([1], -1., 1.))
b = tf.Variable(tf.zeros([1]))
y = W*xs + b
loss = tf.reduce_mean(tf.square(y - ys))
optim = tf.train.GradientDescentOptimizer(ETA)
train = optim.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(EPOCHS):
    sess.run(train)
    print(f'{i}: W: {sess.run(W)}, b: {sess.run(b)}, loss: {sess.run(loss)}')
