from keras.models import Sequential

model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='random_uniform'))

# some other intializers:
# random_normal, zero, and...
# https://keras.io/initializers/

# Activation functions:
# https://keras.io/activations/
