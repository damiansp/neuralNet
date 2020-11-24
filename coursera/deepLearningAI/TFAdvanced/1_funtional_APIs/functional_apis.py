# Examples only -- not functional code
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model


# Sequential model
seq_mod = Sequential([Flatten(input_shape=(28, 28)),
                      Dense(128, activation='relu'),
                      Dense(10, activation='softmax')])

# Functional model
def build_func_mod():
    input = Input(shape=(28, 28))
    flat = Flatten()(input)
    dense = Dense(128, activation='relu')(flat)
    preds = Dense(10, activation='softmax')(dense)
    func_mod = Model(inputs=input, outputs=preds)
    return func_mod

