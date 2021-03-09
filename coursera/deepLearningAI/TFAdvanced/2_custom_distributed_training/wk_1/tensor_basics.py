import tensorflow as tf
from   tensorflow.keras import Sequential
from   tensorflow.keras.layers import Dense


t_var = tf.Variable('Hello', dtype=tf.string)
t_const = tf.constant([1, 2, 3, 4, 5])
print(t_var)
print(t_const)

mod = Sequential([Dense(1, input_shape=(1,))])
print(mod.variables)
