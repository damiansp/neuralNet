from keras.applications.vgg16 import preprocess_input, VGG16
from keras.models import Model
from keras.preprocessing import image

# Pre-built and pre-trained network (VGG16)
base_model = VGG16(weights='imagenet', include_top=True)
for i, layer in enumerate(base_model.layers):
    print(i, layer.name, layer.output_shape)

# Extract features from block4_pool block
model = Model(input=base_model.input,
              output=base_model.get_layer('block4_pool').output)
img_path = 'cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Get features from this block
features = model.predict(x)
