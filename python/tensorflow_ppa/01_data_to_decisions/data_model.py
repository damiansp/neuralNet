import numpy as np
import tensorflow as tf


X = [[2., 4.],
     [6., 8.]]
Y = np.array([[2., 4.],
              [6. , 8.]],
             dtype=np.float32)
Z = tf.constant([[2., 4.],
                 [6., 8.]])
print(type(X))
print(type(Y))
print(type(Z))

tf1 = tf.convert_to_tensor(X, dtype=tf.float32)
tf2 = tf.convert_to_tensor(Y, dtype=tf.float32)
tf3 = tf.convert_to_tensor(Z, dtype=tf.float32)
print(type(tf2), type(tf2), type(tf3))


# Shape
scalar = tf.constant(100)
vector = tf.constant([1, 2, 3, 4, 5])
matrix = tf.constant([[1, 2, 3], [4, 5, 6]])
cube_t  = tf.constant([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]])
print(scalar.get_shape()) # ()
print(vector.get_shape()) # (5,)
print(matrix.get_shape()) # (2, 3)
print(cube_t.get_shape()) # (3, 3, 1)


# Data Type
t1d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
t1d = tf.constant(t1d)
t2d = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)])
t2d = tf.Variable(t2d)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(t1d.get_shape()) # (10,)
    print(sess.run(t1d))   # [ 1 2 3 4 5 6 7 8 9 10]
    print(t2d.get_shape()) # (3, 3)
    print(sess.run(t2d))   # [[1 2 3][4 5 6][7 8 9]]


