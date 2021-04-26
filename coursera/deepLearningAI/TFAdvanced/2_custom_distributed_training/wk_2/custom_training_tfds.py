import tensorflow as tf
import tensorflow.datasets as tfds
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy as Scxe
from tensorflow.keras.optimizers import Adam


N_CLASSES = 10
DENSE_NODES = 64
BATCH = 64
BUFFER = 1024 # amt loaded into memory at a time
EPOCHS = 20


def main():
    train_data = tfds.load('fashion_mnist', split='train')
    test_data = tfds.load('fashion_mnist', split='test')
    train_data = train_data.map(format_image)
    test_data = test_data.map(format_image)
    train = train_data.shuffle(buffer_size=BUFFER).batch(BATCH)
    test = test_data.batch(batch_size=BATCH)
    loss = Scxe()
    optimizer = Adam()
    mod = base_model()
    for epoch in range(EPOCHS):
        loss_train = train_data_for_one_epoch()
        loss_val = perform_validation()
        mean_loss_train = np.mean(loss_train)
        mean_loss_val = np.mean(loss_val)


def format_image(data):
    img = data['image']
    img = tf.reshape(img, [-1])
    img = tf.cast(img, 'float32')
    img = img / 255.
    return img, data['label']


def base_model():
    inputs = Input(shape=(784,), name='clothing')
    x = Dense(DENSE_NODES, activation='relu', name='dense_1')(inputs)
    x = Dense(DENSE_NODES, activation='relu', name='dense_2')(x)
    outputs = Dense(N_CLASSES, activation='softmax', name='predictions')(x)
    mod = Model(inputs=inputs, outputs=outputs)
    return mod


def train_data_for_one_epoch():
    losses = []
    for step, (X_batch_train, y_batch_train) in enumerate(train_dataset):
        logits, loss = apply_gradient(
            optimizer, mod, X_batch_train, y_batch_train)
        losses.append(loss)
    return losses


def apply_gradient(optimizer, mod, X, y):
    with tf.GradientTape() as t:
        logits = mod(X)
        loss = get_loss(y=y, preds=logits)
    grads = t.gradient(loss, mod.trainable_weights)
    optimizer.apply_gradiens(zip(grads, mod.trainable_weights))
    return logts, loss


def perform_validation():
    losses = []
    for X_val, y_val in test:
        val_logits = mod(X_val)
        val_loss = get_loss(y=y_val, preds=val_logits)
        losses.append(val_loss)
    return losses
