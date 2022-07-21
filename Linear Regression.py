"""Simple example of linear regression"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70],
                   [74, 66, 43],
                   [91, 87, 65],
                   [88, 134, 59],
                   [101, 44, 37],
                   [68, 96, 71],
                   [73, 66, 44],
                   [92, 87, 64],
                   [87, 135, 57],
                   [103, 43, 36],
                   [68, 97, 70]],
                  dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119],
                    [57, 69],
                    [80, 102],
                    [118, 132],
                    [21, 38],
                    [104, 118],
                    [57, 69],
                    [82, 100],
                    [118, 134],
                    [20, 38],
                    [102, 120]],
                   dtype='float32')

# Converting the data from numpy to torch
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

# Training dataset
train_ds = TensorDataset(inputs, targets)

# Batch size
batch_size = 5

# Data loader
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# Base model
model = nn.Linear(3, 2)
preds = model(inputs)

# Loss function
loss_fn = F.mse_loss
loss = loss_fn(model(inputs), targets)

# Optimizer with learning rate
opt = torch.optim.SGD(model.parameters(), lr=0.00001)


# Main training loop
def fit(num_epochs, model, loss_fn, opt):
    # Loop
    for epoch in range(num_epochs):
        for xb, yb in train_dl:
            # Utilizing the model
            pred = model(xb)
            # Calculating loss
            loss = loss_fn(pred, yb)
            loss.backward()
            # Using optimizer
            opt.step()
            opt.zero_grad()
        # Printing progress
        if (epoch+1) % 10 == 0:
            print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, loss.item()))

    # Showing the difference between predictions and targets
    print(model(inputs))
    print(targets)


# Calling the function
fit(1000, model, loss_fn, opt)
