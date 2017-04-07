import tensorflow as tf
from read_data import get_minibatch

x = tf.placeholder(tf.float32, name = 'x', shape = [None, 784])
W = tf.Variable(tf.random_uniform([784, 10], -1, 1), name = 'W')
b = tf.Variable(tf.zeros([10]), name = 'biases')
output = tf.matmul(x, W) + b
init_op = tf.initialize_all_variables()
sess = tf.Session()

sess.run(init_op)

feed_dict = { 'x': get_minibatch() }

sess.run(output, feed_dict = feed_dict)
