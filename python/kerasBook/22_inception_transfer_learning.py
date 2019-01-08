from kearas import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import image

# Base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# Add global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(200, activation='softmax')(x)
model = Model(input=base_model.input, output=predictions)

# Freeze pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Now ok to compile
model.compile(optimizer='remsprop', loss='categorical_crossentropy')

model.fit_generator(...)

# Try freezing more (or fewer) layers (i.e., treat as tunable hyperparam)
LAST_FROZEN_LAYER = 172
for layer in model.layers[:LAST_FROZEN_LAYER]:
    layer.trainable = False

for layer in model.layers[LAST_FROZEN_LAYER:]:
    layer.trainable = True


model.compile(optimizer=SGDI(lr=0.0001, momentum=0.9),
              loss='categorical_crossentropy')
model.fit_generator(...)
