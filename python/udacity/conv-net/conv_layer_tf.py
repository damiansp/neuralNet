img_height = 10
img_width = 10
channels = 3
channels_out = 64

filter_height = 5
filter_width = 5


# Conv Layer
X = tf.placeholder(tf.float32, shape=[None, img_height, img_width, channels])
W = tf.Variable(tf.truncated_normal(
    [filter_height, filter_width, channels, channels_out]))
b = tf.Variable(tf.zeros(channels_out))

conv_layer = tf.nn.conv2d(X, W, strides=[1, 2, 2, 1], padding='SAME') + b
conv_layer = tf.nn.relu(conv_layer)

# Max Pooling Layer
conv_layer = tf.nn.max_pool(
    conv_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
