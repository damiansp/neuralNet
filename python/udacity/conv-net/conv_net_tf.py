import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./data/', one_hot=True, reshape=False)

ETA = 0.00001
EPOCHS = 10
BATCH_SIZE = 128
N_CLASSES = 10
DROPOUT = 0.75

test_valid_size = 256
weights = {'Wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
           'Wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
           'Wd':  tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
           'W_out': tf.Variable(tf.random_normal([1024, N_CLASSES]))}
biases = {'bc1': tf.Variable(tf.random_normal([32])),
          'bc2': tf.Variable(tf.random_normal([64])),
          'bd':  tf.Variable(tf.random_normal([1024])),
          'b_out': tf.Variable(tf.random_normal([N_CLASSES]))}

def conv2d(X, W, b, strides=1, activation=tf.nn.relu):
    X = tf.nn.conv2d(X, W, strides=[1, strides, strides, 1], padding='SAME') + b
    return activation(X)

def maxpool2d(X, k=2):
    return tf.nn.max_pool(
        X, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net(X, W, b, dropout):
    # L1 28x28x1 -> 14x14x32
    conv1 = conv2d(X, W['Wc1'], b['bc1'])
    conv1 = maxpool2d(conv1)

    # L2 14x14x32 -> 7x7x64
    conv2 = conv2d(conv1, W['Wc2'], b['bc2'])
    conv2 = maxpool2d(conv2)

    # L3 FC: 7x7x64 -> 1024
    fc1 = tf.reshape(conv2, [-1, W['Wd'].get_shape().as_list()[0]])
    fc1 = tf.matmul(fc1, W['Wd']) + b['bd']
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    # Out 1024 -> 10
    out = tf.matmul(fc1, W['W_out']) + b['b_out']
    return out

# Graph inputs
X = tf.placeholder(tf.float32, [None, 28, 28, 1]) # 28x28 B/W image
y = tf.placeholder(tf.float32, [None, N_CLASSES])
keep_prob = tf.placeholder(tf.float32)

# Model
logits = conv_net(X, weights, biases, keep_prob)

# Loss/optimizer
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=ETA).minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Init
init = tf.global_variables_initializer()


# Launch graph
with tf.Session() as s:
    s.run(init)
    acc = []

    for epoch in range(EPOCHS):
        for batch in range(mnist.train.num_examples // BATCH_SIZE):
            batch_X, batch_y = mnist.train.next_batch(BATCH_SIZE)
            s.run(optimizer,
                  feed_dict={X: batch_X, y: batch_y, keep_prob: DROPOUT})

            # Batch loss, accuracy
            loss = s.run(cost,
                         feed_dict={X: batch_X, y: batch_y, keep_prob: 1.})
            valid_acc = s.run(
                accuracy,
                feed_dict={X: mnist.validation.images[:test_valid_size],
                           y: mnist.validation.labels[:test_valid_size],
                           keep_prob: 1.})

            acc.append(valid_acc)
            print('Epoch {:>2}, Batch {:>3} - Loss: {:>10.4f} '
                  'Valid. Acc: {:.6f}'.format(
                      epoch + 1, batch + 1, loss, valid_acc))

    # Test Accuracy
    test_acc = s.run(accuracy,
                     feed_dict={X: mnist.test.images[:test_valid_size],
                                y: mnist.test.labels[:test_valid_size],
                                keep_prob: 1.})
    print('Test Accuracy: {}'.format(test_acc))
    plt.plot(acc, 'r-')
    plt.show()
