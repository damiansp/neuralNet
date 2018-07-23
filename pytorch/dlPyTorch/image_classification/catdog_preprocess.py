import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder


def imshow(inp):
    inp = inp.np().transpose([1, 2, 0])
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std*inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    
simple_transform = transforms.Compose(
    [transforms.Resize([224, 224]),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
train = ImageFolder('../data/dogs_and_cats/train/', simple_transform)
valid = ImageFolder('../data/dogs_and_cats/valid/', simple_transform)

plt.imshow(train[50][0])
plt.show()
