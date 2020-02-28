import tensorflow as tf


x = tf.constant(8)
y = tf.constant(9)
z = tf.multiply(x, y)

s = tf.Session()
out_z = s.run(z)
s.close()

print(f'8 x 9: {out_z}')



# Alternately:
with tf.Session() as s:
    x = tf.placeholder(tf.float32, name='x')
    y = tf.placeholder(tf.float32, name='y')
    z = tf.multiply(x, y)
    z_out = s.run(z, feed_dict={x: 7, y: 8})
print(f'7 x 8: {z_out}')
