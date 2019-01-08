import numpy as np
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dropout, Reshape
from keras.layers.convolutional import ZeroPadding2D
from keras.models import Sequential


def test_layer(layer, x):
    layer_config = layer.get_config()
    layer_config['input_shape'] = x.shape
    layer = layer.__class__.from_config(layer_config)

    mod = Sequential()
    mod.add(layer)
    mod.compile('rmsprop', 'mse')
    x_ = np.expand_dims(x, axis=0)
    return mod.predict(x_)[0]


x = np.random.randn(10, 10)
layer = Dropout(0.5)
y = test_layer(layer, x)
assert(x.shape == y.shape)

x = np.random.randn(10, 10, 3)
layer = ZeroPadding2D(padding=(1, 1))
y = test_layer(layer, x)
assert(x.shape[0] + 2 == y.shape[0])
assert(x.shape[1] + 2 == y.shape[1])

x = np.random.randn(10, 10)
layer = Reshape((5, 20))
y = test_layer(layer, x)
assert(y.shape == (5, 20))


# Implement Local Response Normalization:
# LRN(x[i]) = x[i] / (k + (alpha/n) * sum(x)) ** beta

class LocalResponseNormalization(Layer):
    def __init__(self, n=5, alpha=0.0005, beta=0.75, k=2, **kwargs):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.k = k
        super(LocalResponseNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = input_shape
        super(LocalResponseNormalization, self).build(input_shape)

    def call(self, x, mask=None):
        if K.image_dim_ordering == 'th':
            _, f, r, c = self.shape # _ for batch size
        else:
            _, r, f, c = self.shape
        squared = K.square(x)
        pooled = K.pool2d(squared,
                          (self.n, self.n),
                          strides=(1, 1),
                          padding='same',
                          pool_mode='avg')
        ax = 1 if K.image_dim_ordering == 'th' else 3
        summed = K.sum(pooled, axis=ax, keepdims=True)
        averaged = self.alpha * K.repeat_elements(summed, f, axis=ax)
        denom = K.pow(self.k + averaged, self.beta)
        return x / denom

    def get_output_shape_for(self, input_shape):
        return input_shape


x = np.random.randn(225, 225, 3)
layer = LocalResponseNormalization()
y = test_layer(layer, x)
assert(x.shape == y.shape)
        
