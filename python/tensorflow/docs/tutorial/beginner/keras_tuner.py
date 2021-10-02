import keras_tuner as kt
from   tensorflow import keras
from   tensorflow.keras import Sequential
from   tensorflow.keras.callbacks import EarlyStopping
from   tensorflow.keras.layers import Dense, Flatten
from   tensorflow.keras.losses import SparseCategoricalCrossentropy
from   tensorflow.keras.optimizers import Adam


INPUT_SHAPE = (28,28) # fashion mnist image file dims
N_CLASSES = 10
EPOCHS = 50


def main():
    (X_train, y_train), (X_test, y_test) = (
        keras.datasets.fashion_mnist.load_data())
    X_train, X_test = [normalize(X) for X in [X_train, X_test]]
    tuner, best_hps = get_best_hps(X_train, y_train)
    mod = tuner.hypermodel.build(best_hps)
    hypermod, best_epoch = get_hypermod(X_train, y_train, mod, tuner, best_hps)
    evaluate(hypermod, X_train, y_train, X_test, y_test, best_epoch)

    
def normalize(X):
    X = X.astype('float32') / 255.
    return X


def get_best_hps(X_train, y_train):
    print('Finding best hyperparameters...')
    tuner = kt.Hyperband(build_mod,
                         objective='val_accuracy',
                         max_epochs=10,
                         factor=3,
                         direcory='.',
                         project_name='kt_intro')
    stop_early = EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(X_train,
                 y_train,
                 epochs=EPOCHS,
                 validation_split=0.2,
                 callbacks=[stop_early])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print('Best hyperparams found:\n'
          f'  Dense units: {best_hps.get("units")}\n'
          f'  Learning rate: {best_hps.get("learning_rate")}')
    return tuner, best_hps


def build_mod(hp):
    print('Building model...')
    mod = Sequential()
    mod.add(Flatten(input_shape=INPUT_SHAPE))
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    mod.add(Dense(units=hp_units, activation='relu'))
    mod.add(Dense(N_CLASSES))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    mod.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                loss=SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return mod


def get_hypermod(X_train, y_train, mod, tuner, best_hps):
    print('Getting hypermodel...')
    history = mod.fit(X_train, y_train, epcohs=EPOCHS, validation_split=0.2)
    val_accs = history.history['val_accuracy']
    best_epoch = val_accs.index(max(val_accs)) + 1
    print('Best epoch:', best_epoch)
    hypermod = tuner.hypermodel.build(best_hps)
    return hypermod, best_epoch


def evaluate(hypermod, X_train, y_train, X_test, y_test, best_epoch):        
    hypermod.fit(X_train, y_train, epochs=best_epoch, validation_split=0.2)
    eval_res = hypermod.evaluate(X_test, y_test)
    print('[test loss, test acc]:', eval_res)


if __name__ == '__main__':
    main()
    
