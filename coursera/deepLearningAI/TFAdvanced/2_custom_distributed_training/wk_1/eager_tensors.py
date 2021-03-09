import numpy as np
import tensorflow as tf
from   tensorflow.keras.layers import Layer


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


v = tf.Variable(0.)
print(v + 1) # Tensor(1.0, ...)

v.assign_add(1)
print(v)                      # <tf.Variable 'Variable:0'..., numpy=1.0> 
print(v.read_value().numpy()) # 1.0


class MyLayer(Layer):
    def __init__(self):
        super().__init__()
        self.my_var = tf.Variable(100)
        self.my_var_list = [tf.Variable(x) for x in range(2)]


m = MyLayer()
print([var.numpy() for var in m.variables]) # [100, 0, 1]


t = tf.constant([1, 2, 3])
print(t) 

t2 = tf.cast(t, dtype=tf.float32)
print(t2)
