import numpy as np
import scipy.misc
from keras.models import model_from_json
from keras.optimizers import SGD

# Load model
model_architecture = 'cifar10_architecture.json'
model_weights = 'cifar10_weights.h5'
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)

# Load images
img_names = ['cat.jpg', 'dog.jpg']
imgs = [
    np.transpose(
        scipy.misc.imresize(
            scipy.misc.imread(img_name), (32, 32)),
        (1, 0, 2))\
    .astype('float32')
    for img_name in img_names]
imgs = np.array(imgs) / 255

# Compile model
model.compile(
    loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])

# Predict
preds = model.predicts_classes(imgs)
print(predictions) # will give numeric version of labels
