import matplotlib.pyplot as plt
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
boston_tensor = torch.from_numpy(boston.data)
print('boston_tensor:', boston_tensor.size())
print(boston_tensor[:2])

# 3D Tensors
me = np.array(Image.open('images/me.jpg'))#.resize((224, 224))) # 825, 743, 3
me_tensor = torch.from_numpy(me)
print(me_tensor.size())
plt.imshow(me)
plt.show()

# Slicing Tensors
sales = torch.FloatTensor(
    [1000., 323.2, 333.4, 444.5, 1000., 323.2, 333.4, 444.5])
print(sales[:5])
print(sales[:-5])
plt.imshow(me_tensor[:, :, 0].numpy())
plt.show()
plt.imshow(me_tensor[:, :, 1].numpy())
plt.show()
plt.imshow(me_tensor[:, :, 2].numpy())
plt.show()


