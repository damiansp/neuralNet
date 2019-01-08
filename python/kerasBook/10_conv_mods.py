from keras.layers.convolutional import (
    Conv1D, Conv2D, MaxPooling1D, MaxPooling2D)

Conv1D(filters,
       kernel_size,
       strides=1,
       padding='valid',
       dilation_rate=1,
       activation=None,
       use_bias=True,
       kernel_initializer='glorot_uniform',
       bias_initializer='zeros',
       kernel_regularizer=None,
       bias_regularizer=None,
       activity_regularizer=None,
       kernel_constraint=None,
       bias_constraint=None)
Conv2D(filters,
       kernel_size,
       strides=(1, 1),
       padding='valid',
       data_format=None,
       dilation_rate=(1, 1),
       activation=None,
       use_bias=True,
       kernel_initializer='glorot_uniform',
       bias_initializer='zeros',
       kernel_regularizer=None,
       bias_regularizer=None,
       activity_regularizer=None,
       kernel_constraint=None,
       bias_constraint=None)
MaxPooling1D(pool_size=2, strides=None, padding='valid')
MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
