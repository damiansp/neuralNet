import tensorflow.datasets as tfds
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D,
    MaxPool2D)

class IdentityBlock(Model):
    def __init__(self, filters, kernel_size, name=''):
        super().__init__(name=name)
        self.conv1 = Conv2D(filters, kernel_size, padding='same')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(filters, kernel_size, padding='same')
        self.bn2 = BatchNormalization()
        self.act = Activation('relu')
        self.add = Add()

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.add([x, input_tensor])
        x = self.act(x)
        return x

# X  -> 3x3Conv -> BN -> ReLU -> 3x3Conv -> BN
#  \                                          \
#   \                                          + -> ReLU -> out
#    \________________________________________/


class ResNet(Model):
    def __init__(self, n_classes):
        super().__init__()
        self.conv = Conv2D(64, 7, padding='same')
        self.bn = BatchNormalization()
        self.act = Activation('relu')
        self.pool = MaxPool2D((3, 3))
        self.id1a = IdentityBlock(64, 3)
        self.id1b = IdentityBlock(64, 3)
        self.global_pool = GlobalAveragePooling2D()
        self.classifier = Dense(n_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.id1a(x)
        x = self.id1b(x)
        x = self.global_pool(x)
        return self.classifier(x)

# X -> 7x7Conv -> BN -> 3x3MxPool -> ID -> ID -> GlobalAvgPool


resnet = ResNet(10)
resnet.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
dataset = tfds.load('mnist', split=tfds.Split.TRAIN)
dataset = dataset.map(preprocess).batch(32)
resnet.fit(dataset, epochs=10)
