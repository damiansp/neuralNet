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

tensor_3d = np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                      [[2, 3, 4], [5, 6, 7], [8, 9, 0]],
                      [[4, 5, 6], [7, 8, 9], [0, 1, 2]]])
tensor_3d = tf.convert_to_tensor(tensor_3d, dtype=tf.float64)
with tf.Session() as sess:
    print(tensor_3d.get_shape()) # (3, 3, 3)
    print(sess.run(tensor_3d))


# Variables
value = tf.Variable(0, name='value')
one = tf.constant(1)
new_value = tf.add(value, one)
update_value = tf.assign(value, new_value)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(value)) # 0
    for _ in range(5):
        sess.run(update_value)
        print(sess.run(value)) # 1, 2, 3, 4, 5


# Fetching
const_A = tf.constant([100.])
const_B = tf.constant([300.])
const_C = tf.constant([3.])
sum_ = tf.add(const_A, const_B)
prod_ = tf.multiply(const_A, const_C)
with tf.Session() as sess:
    result = sess.run([sum_, prod_])
    print(result) # [400.], [300.]


# Feeds and Placeholders
a = 3
b = 2
x = tf.placeholder(tf.float32, shape=(a, b))
y = tf.add(x, x)
data = np.random.rand(a, b)
sess = tf.Session()
print(sess.run(y, feed_dict={x: data}))
sess.close()
