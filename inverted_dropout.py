import numpy as np

network.p = 0.5

def train_step(network, X):
    # Forward pass for 3-layered NN
    l1 = np.maximum(0, np.dot(network.W1, X) + network.b1)
    dropout1 = (np.random.rand(l1.shape) < network.p) / network.p
    l1 *= dropout1

    l2 = np.maximum(0, np.dot(network.W2, l1) + network.b2)
    dropout2 = (np.random.rand(l2.shape) < network.p) / network.p
    l2 *= dropout2

    output = np.dot(network.W3, l2) + network.b3

    # back pass: Compute gradients (omitted)...
    # update params (omitted) ...

def predict(network, X):
    l1 = np.maximum(0, np.dot(network.W1, X) + network.b)
    l2 = np.maximum(0, np.dot(network.W2, l1) + network.b1)
    output = np.dot(network.W3, l2) + network.b3
    return output
                        

