from datetime import datetime

import imageio
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop


mod = Sequential([Dense(units=1, activation='linear', input_dim=(784,))])
mod.compile(
    optimizer=RMSprop(lr=0.1), loss='mean_squared_error', metrics=['mae'])

                 
class MyCallback(Callback):
    def on_train_batch_begin(self, batch, logs=None):
        print(f'Training: batch {batch} starting at {datime.now().time()}')

    def on_train_batch_end(self, batch, logs=None):
        print(f'Training: batch {batch} ended at {datime.now().time()}')


my_cb = MyCallback()
mod.fit(X_train, y_train, batch_size=64, epochs=5, verbose=0, callbacks=[my_cb])




# more complex example
class OverfitDetectorCB(Callback):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    # does it execute per batch or per epoch??
    def on_train_batch_begin(self, epoch, logs=None):
        ratio = logs['val_loss'] / logs['loss']
        print(f'Epoch {epoch}: Val/Train Loss Ratio: {ratio:.2f}')
        if ratio > threshold:
            print('Threshold reached. Stopping...')
            self.model.stop_training = True

#...
mod.fit(..., callbacks=[OverfitDetectorCB(threshold=1.3)])



# CNN Vis sample
class VisCallback(Callback):
    def __init__(slef, inputs, ground_truth, display_freq=10, n_samples=10):
        self.inputs = inputs
        self.ground_truth = ground_truth
        self.images = []
        self.display_freq = display_freq
        self.n = n_samples

    def on_epoch_end(self, epoch, logs=None):
        idx = np.random.choice(len(self.inputs), size=self.n)
        X_test, y_test = self.inputs[idx], self.ground_trugh[idx]
        preds = np.argmax(self.model.predict(X_test), axis=1)
        display_digits(X_test, preds, y_test, epoch)
        # save fig
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        self.images.append(np.array(img))
        if epoch % self.display_freq == 0:
            plt.show()

    def on_train_end(self, logs=None):
        imageio.mimsave('animation.gif', self.images, fps=1)


mod.fit(..., callbacks=[VisCallback(X_test, y_test)])
