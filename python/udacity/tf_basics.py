import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from pprint import pprint

# Mini-batching:
def batches(batch_size, X, Y):
    assert len(X) == len(Y)

    n = len(X)
    n_batches = int(np.ceil(n / batch_size))
    n_per_batch = n // n_batches
    batches = []

    for i in range(n_batches):
        start = i * n_per_batch
        end = max(start + n_per_batch, n - 1)
        batch = [X[start:end], Y[start:end]]
        batches.append(batch)

    return batches

# Test
X = [['F11','F12','F13','F14'],
     ['F21','F22','F23','F24'],
     ['F31','F32','F33','F34'],
     ['F41','F42','F43','F44']]
Y = [['L11','L12'],
     ['L21','L22'],
     ['L31','L32'],
     ['L41','L42']]

pprint(batches(3, X, Y))



# Epochs
def print_epoch_stats(epoch, sess, last_X, last_Y):
    current_cost = sess.run(cost, feed_dict={X: last_X, Y: last_Y})
    valid_accuracy = sess.run(accuracy, feed_dict={X: valid_X, Y: valid_Y})
    print('Epoch: {:<4} - Cost: {:<8.3} Valid Acc: {:<5.3}'.format(
        epoch,
        current_cost,
        valid_accuracy))

n_input = 28 * 28 # MNIST image dims
n_classes = 10    # digits 0 - 9
mnist = input_data.read_data_sets('data/mnist', one_hot=True)
train_X = mnist.train.images
valid_X = mnist.validation.images
test_X  = mnist.test.images
train_Y = mnist.train.labels.astype(np.float32)
valid_Y = mnist.validation.labels.astype(np.float32)
test_Y  = mnist.test.labels.astype(np.float32)

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_classes])
W = tf.Variable(tf.random_normal([n_input, n_classes]))
b = tf.Variable(tf.zeros([n_classes]))
logits = tf.matmul(X, W) + b

eta = tf.placeholder(tf.float32)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                              labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=eta).minimize(cost)
correct = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()

BATCH = 128
EPOCHS = 10
ETA = 0.001
train_batches = batches(BATCH, train_X, train_Y)

with tf.Session() as s:
    s.run(init)

    # Train
    for epoch in range(EPOCHS):
        for batch_X, batch_Y in train_batches:
            s.run(optimizer, feed_dict={X: batch_X,
                                        Y: batch_Y,
                                        eta: ETA})
        print_epoch_stats(epoch, s, batch_X, batch_Y)

    test_accuracy = s.run(accuracy, feed_dict={X: test_X, Y: test_Y})

print('Test Accuracy: {}'.format(test_accuracy))
