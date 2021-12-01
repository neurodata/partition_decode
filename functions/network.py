import numpy as np
import torch
from torch import nn

import os

## Network functions

# Model
class Net(nn.Module):
    """DeepNet class
    A deep net architecture with `n_hidden` layers,
    each having `hidden_size` nodes.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_size=10,
        n_hidden=2,
        activation=torch.nn.ReLU(),
        bias=False,
        penultimate=False,
        bn=False,
    ):
        super(Net, self).__init__()

        module = nn.ModuleList()
        module.append(nn.Linear(in_dim, hidden_size, bias=bias))

        for ll in range(n_hidden):
            module.append(activation)
            if bn:
                module.append(nn.BatchNorm1d(hidden_size))
            module.append(nn.Linear(hidden_size, hidden_size, bias=bias))

        if penultimate:
            module.append(activation)
            if bn:
                module.append(nn.BatchNorm1d(hidden_size))
            module.append(nn.Linear(hidden_size, 2, bias=bias))
            hidden_size = 2

        module.append(activation)
        if bn:
            module.append(nn.BatchNorm1d(hidden_size))
        module.append(nn.Linear(hidden_size, out_dim, bias=bias))

        self.sequential = nn.Sequential(*module)

    def forward(self, x):
        return self.sequential(x)


# functions
def weight_reset(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        m.reset_parameters()


def train_model(model, train_x, train_y, multi_label=False, verbose=False):
    """
    Train the model given the training data
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = torch.nn.BCEWithLogitsLoss()

    losses = []

    for step in range(1000):
        optimizer.zero_grad()
        outputs = model(train_x)
        if multi_label:
            train_y = train_y.type_as(outputs)

        loss = loss_func(outputs, train_y)
        trainL = loss.detach().item()
        if verbose and (step % 500 == 0):
            print("train loss = ", trainL)
        losses.append(trainL)
        loss.backward()
        optimizer.step()

    return losses


def get_model(
    hidden_size=20,
    n_hidden=5,
    in_dim=2,
    out_dim=1,
    penultimate=False,
    use_cuda=True,
    bn=False,
):
    """
    Initialize the model and send to gpu
    """
    in_dim = in_dim
    out_dim = out_dim  # 1
    model = Net(
        in_dim,
        out_dim,
        n_hidden=n_hidden,
        hidden_size=hidden_size,
        activation=torch.nn.ReLU(),
        bias=True,
        penultimate=penultimate,
        bn=bn,
    )

    if use_cuda:
        model = model.cuda()

    return model
