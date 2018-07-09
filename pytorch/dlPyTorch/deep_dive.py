from torch.autograd import Variable
from torch.nn import Linear


inp = Variable(torch.randn(1, 10))
layer1 = Linear(in_features=10, out_features=5, bias=True)
layer1(inp)

layer2 = Linear(5, 2)
layer2(layer1(inp))


print(layer1.weight)
print(layer1.bias)
