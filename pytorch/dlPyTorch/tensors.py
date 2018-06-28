from glob import glob
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


# 4D Tensors (+batch)
faces = glob('/Users/dsatterthwaite/repos/scripts-analytics/faces/images/raw/'
             + '*.jpg')
# To np arrays
face_imgs = np.array([np.array(Image.open(face).resize([224, 224]))
                      for face in faces[:64]])
face_imgs = face_imgs.reshape(-1, 224, 224, 3)
face_tensor = torch.from_numpy(face_imgs)
print(face_tensor.size())


# Tensors on GPU
a = torch.rand(2, 2)
b = torch.rand(2, 2)
c = a + b
d = torch.add(a, b)
a.add_(b) # in place

print(a*b)
print(a.mul(b))
print(a.mul_(b)) # in place

# Cuda
a = torch.rand(10000, 10000)
b = torch.rand(10000, 10000)
a.matmul(b)

#a = a.cuda()
#b = b.cuda()
#a.matmul(b)
