import torch
from torch.autograd import Variable

# Variables
x = Variable(torch.ones(2, 2), requires_grad=True)
y = x.mean()
y.backward()
print(x.grad)
print(x)
print(x.data)
print(y)
print(y.grad_fn)
