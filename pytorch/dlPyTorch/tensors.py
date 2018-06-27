import numpy as np
import sklearn.datasets
import torch

from PIL import Image

boston = sklearn.datasets.load_boston()

# Scalars
x = torch.rand(10)
print(x, x.size())

# Vectors
vec = torch.FloatTensor([23, 24, 24.5, 26, 27.2, 23.0])
print(vec, vec.size())

# Matrices
#boston_tensor = torch.from_numpy(boston.data)
#print('boston_tensor:', boston_tensor.size())
#print(boston_tensor[:2])

# 3D Tensors
#panda = np.array(Image.open('panda.jpg').resize((224, 224)))
#panda_tensor = torch.from_numpy(panda)
#panda_tensor.size()
#plt.imshow(panda)

# Slicing Tensors
sales = torch.FloatTensor(
    [1000., 323.2, 333.4, 444.5, 1000., 323.2, 333.4, 444.5])
print(sales[:5])
print(sales[:-5])
plt.imshow(panda_tensor
