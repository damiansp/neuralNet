from keras.layers import Activation, Dense, Input
from keras.models import Model, Sequential

# Sequential---------------------------------------------------------------
mod = Sequential([Dense(32, input_dim=784),
                  Activation('sigmoid'),
                  Dense(10),
                  Activation('softmax')])
mod.compile(loss='categorical_crossentropy', optimizer='adam')


# Functional---------------------------------------------------------------
inputs = Input(shape=[784,])
X = Dense(32)(inputs)
X = Activation('sigmoid')(X)
X = Dense(10)(X)
predictions = Activation('softmax')(X)
mod = Model(inputs=inputs, outputs=predictions)
mod.compile(loss='categorical_crossentropy', optimizer='adam')

# preds = \
#    Activation('softmax')(Dense(10)(Activation('sigmoid')(Dense(32)(inputs))))


# The following network architectures MUST use the functional API
# - Models with multiple in/outputs
# - Models composed of multiple submodels
# - Models sharing layers
# e.g.:
mod = Model(inputs=[input1, input2], outputs=[output1, output2])
