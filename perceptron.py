import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
import torch.optim as optim

inputDim = 2
losses = []
epochs = 100
eta = 0.01

# choose learning method
methods = ['backward', 'grad', 'optimizer']
method = methods[0]

# Read data from file
data = pd.read_csv('data.csv')
labels = torch.tensor(data['target'].values, dtype=torch.float32)
data = torch.tensor(data[['x', 'y']].values, dtype=torch.float32)

numpt = data.size(0)

weights = torch.zeros(inputDim, dtype=torch.float32, requires_grad=True)
bias = torch.zeros(1, requires_grad=True)

if method == 'optimizer':
    optimizer = optim.SGD((weights, bias),lr=eta)

for epoch in range(epochs):
    total_loss = 0
    
    for idx in range(numpt):
        # take current input
        X = data[idx,:]
        y = labels[idx]
        

        # compute output and loss
        out = torch.add(torch.dot(weights, X), bias)
        loss = torch.max(torch.tensor(0, dtype=torch.float32), -1 * out * y)
        total_loss += loss.item()
        
        if method == 'grad':
            gradw = torch.autograd.grad(loss, weights, retain_graph=True)
            gradb = torch.autograd.grad(loss, bias, retain_graph=True)
            with torch.no_grad():
                weights -= eta * gradw[0]
                bias -= eta * gradb[0]
        
        
        elif method == 'backward':      
            # backpropagation
            loss.backward()
         
            # compute accuracy and update parameters
            with torch.no_grad():
                weights -= eta * weights.grad
                bias -= eta * bias.grad
                # reset gradient to zero
                weights.grad.zero_()
                bias.grad.zero_()
                
                
        elif method == 'optimizer':
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    losses.append(total_loss / numpt)
    print(losses[-1])


# plot points, hyperplane and learning curve
plt.figure()
plt.scatter(data[:,0].numpy(), data[:,1].numpy(), c=labels.numpy())
xr = np.linspace(0, 20, 10)
yr = (-1 / weights[1].item()) * (weights[0].item() * xr  + bias.item())
plt.plot(xr, yr,'-')
plt.show()

plt.figure()
plt.plot(losses, '-')
plt.show()
