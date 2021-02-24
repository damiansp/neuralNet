import numpy as np
import tensorflow as tf


print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))
print(tf.square(2) + tf.square(3))

x = np.arange(25)
x = tf.constant(x)
xsq = tf.square(x)
print(xsq)

x2 = tf.reshape(xsq, (5, 5))
print(x2)

xf = tf.cast(x2, tf.float32)
print(xf)


y = tf.constant(2, dtype=tf.float32)
res = tf.multiply(xf, y)
print(res)

y = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
res = xf + y
print(res)
