class CNNResidual(Layer):
    def __init__(self, layers, filters, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [Conv2D(filters, (3, 3), activation='relu')
                       for _ in range(layers)]

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        return inputs + x


class DenseResidual(Layer):
    def __init__(self, layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [Dense(n_neurons, activation='relu')
                       for _ in range(layers)]

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        return inputs + x


class MyResidual(Model):
    def __init__(self, **kwargs):
        self.h1 = Dense(30, activation='relu')
        self.block1 = CNNResidual(2, 32)
        self.block2 = DenseResidual(2, 64)
        self.out = Dense(1)

    def call(self, inputs):
        x = self.h1(inputs)
        x = slef.block1(x)
        for _ in range(4):
            x = self.block2(x)
        return self.out(x)
