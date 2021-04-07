import tensorflow as tf


class Model():
    def __init__(self):
        self.W = tf.Variable(5.)
        self.b = tf.Variable(0.)

    def __call__(self, x):
        return self.W*x + self.b


def mse(y, pred):
    return tf.reduce_mean(tf.square(y - pred))


def train(mod, inputs, outputs, eta):
    with tf.GradientTape() as tape:
        loss = mse(outputs, mod(inputs))
    dW, db = tape.gradient(loss, [mod.W, mod.b])
    mod.W.assign_sub(eta * dW)
    mod.b.assign_sub(eta * db)

    
TRUE_W = 3.
TRUE_B = 2.
N = 1000
EPOCHS = 20
ETA = 0.1

random_xs = tf.random.normal(shape=[N])
ys = (TRUE_W * random_xs) + TRUE_B
mod = Model()

for e in range(EPOCHS):
    train(mod, random_xs, ys, eta=ETA)
    print(mod.W, mod.b)
