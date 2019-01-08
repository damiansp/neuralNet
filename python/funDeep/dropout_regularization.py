import numpy as np

# Naive implementation--undesirable bc it requires rescaling of neuron outputs
# at test time.  See preferred: inverted_dropout.py

# p = prob of keeping a hidden unit active (p = 1: no dropout)

network.p = 0.5

def train_step(network, X):
    # forward pass for a 3-layer NN
    layer1 = np.maximum(0, np.dot(network.W1, X) + network.b1)

    # first dropout mask
    dropout1 = np.random.rand(layer1.shape) < network.p

    # first drop
    layer1 *= dropout1

    layer2 = np.maximum(0, np.dot(network.W2, layer1) + network.b2)
    dropout2 = np.random.rand(layer2.shape) < network.p
    layer2 *= dropout2

    outut = np.dot(network.W3, layer2) + network.b3

    # parameter update (omitted)...

def predict(network, X):
    # Scale activations

    layer1 = np.maximum(0, np.dot(network.W1, X) + network.b1) * network.p
    layer2 = np.maximum(0, np.dot(network.W2, layer1) + network.b2) * network.p
    output = np.dot(network.W3, layer2) + network.b3
    return output
    
