import numpy as np
import tensorflow as tf


x = 2
xsq = tf.square(x)
print(f'x^2 = {xsq}')


a = tf.constant([[1, 2], [3, 4]])
print(tf.add(a, 1))
print(a ** 2)


a = tf.constant(5)
b = tf.constant(3)
print(np.multiply(a, b))

nda = np.ones([3, 3])
tensor = tf.multiply(nda, 3)
print(tensor)
print(tensor.numpy())
