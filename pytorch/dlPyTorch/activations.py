import torch
from torch.autograd import Variable
from torch.nn import ReLU


sample_data = Variable(torch.Tensor([[1, 2, -1, -1]]))
relu = ReLU()
print(relu(sample_data))
                       
