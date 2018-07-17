import torch
from torch import nn
from torch.autograd import Variable

loss = nn.MSELoss()
input = Variable(torch.randn(3, 5), requires_grad=True)
target = Variable(torch.randn(3, 5))
output = loss(input, target)
output.backward()
print(output)


def cross_entropy(true_label, prediction):
    if true_label == 1:
        return -log(prediction)
    return -log(1 - prediction)

loss = nn.CrossEntropyLoss()
target = Variable(torch.LongTensor(3).random_(5))
output = loss(input, target)
output.backward()
print(output)
