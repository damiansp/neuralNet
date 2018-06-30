import numpy as np
import torch
from torch.autograd import Variable


def get_data():
    train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59,
                          2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
    train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596,
                          2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94,
                          1.3])
    dtype = torch.FloatTensor
    X = Variable(torch.from_numpy(train_X).type(dtype), requires_grad=False)\
        .view(17, 1)
    y = Variable(torch.from_numpy(train_Y)type(dtype), requires_grad=False)
    return X, y


def get_weights():
    W = Variable(torch.randn(1), requires_grad=True)
    b = Variable(torch.randn(1), requires_grad=True)
    return W, b



        
