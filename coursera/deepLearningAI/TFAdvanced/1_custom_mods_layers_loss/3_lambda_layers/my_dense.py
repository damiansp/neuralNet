# Extending TF's Layer class example
class SimpleDense(Layer):
    def __init__(self, units=32, activation=None):
        super(SimpleDense, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        # Create the state of the layer (weights)
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            name='weights',
            initial_value=w_init(shape=(input_shape[-1], self.units),
                                 dtype='float32'),
            trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            name='bias',
            initial_value=b_init(shape=(slef.units,), dtype='float32'),
            trainable=True)

    def call(self, inputs):
        # Computations in forward pass
        return self.activation(tf.matmul(inputs, self.w) + self.b)


my_dense = SimpleDense(units=1)
X = tf.ones((1, 1))
y = my_dense(X)
print(my_dense.variables)
