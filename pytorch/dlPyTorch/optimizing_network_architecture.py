from torch import optim

optimizer = optim.SGD(meld.parameters(), lr=0.01)

for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()

    
