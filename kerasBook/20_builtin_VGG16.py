import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.vgg16 import VGG16 # large image recognition model
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import image

# Prebuild model with pre-trained weights on imagenet
model = VGG16(weights='imagenet', include_top=True)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

# Resize to VGG16 format
img = cv2.resize(cv2.imread('steam-locomotive.jpg'), (224, 224))
img = np.expand_dims(img, axis=0)

# Predict
pred = model.predict(img)
plt.plot(pred.ravel())
plt.show()

print(np.argmax(pred))
# plot and value should be 820 (steaming train) with little support (~0.0) for
# other values
