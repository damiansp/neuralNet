from torch import nn

class BasicNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BasicNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def __forward__(self, input):
        out = self.layer1(input)
        out = nn.ReLU(out)
        out = self.layer2(out)
        return out
