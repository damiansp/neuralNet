from torchvision import transforms
from torchvision.datasets import ImageFolder

simple_transform = transforms.Compose(
    [transforms.Resize([224, 224]),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
train = ImageFolder('../data/dogs_and_cats/train/', simple_transform)
valid = ImageFolder('../data/dogs_and_cats/valid/', simple_transform)
