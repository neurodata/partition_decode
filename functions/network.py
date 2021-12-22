import numpy as np
import torch
from torch import nn

import os

## Network functions

# Model
class Net(nn.Module):
    """
    A class for a deep neural net architecture.

    Parameters
    ----------
    in_dim: int
        Input dimension.

    out_dim: int
        Output dimension.

    hidden_size: int, default = 10
        Number of nodes in every hidden layer.

    n_hidden: int, default = 2
        Number of hidden layers

    activation: ACTIVATION, default = torch.nn.ReLU()
        Activation function to be used by the hidden layers.

    bias: bool, default = False
        A boolean indicating if a bias shall be added.

    bn: bool, default = False
        A boolean indicating if batch norm shall be applied.

    """

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_size=10,
        n_hidden=2,
        activation=torch.nn.ReLU(),
        bias=False,
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

        module.append(activation)
        if bn:
            module.append(nn.BatchNorm1d(hidden_size))
        module.append(nn.Linear(hidden_size, out_dim, bias=bias))

        self.sequential = nn.Sequential(*module)

    def forward(self, x):
        return self.sequential(x)


## functions


def weight_reset(m):
    """
    Reinitializes parameters of a model [m] according to default initialization scheme.
    """
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        m.reset_parameters()


def train_model(model, train_x, train_y, multi_label=False, verbose=False):
    """
    Performs training of a model given training data.

    Parameters
    ----------

    model : Net
        A deep neural net model to be trained

    train_x : Tensor
        Training features

    train_y: Tensor
        Training labels

    multi_label: bool, default = False
        A boolean indicating if it is a multi-label classification

    verbose: bool, default = False
        A boolean indicating the production of detailed logging information during training

    Returns
    -------
    losses : Tensor
        The accumulated losses during training.
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
    in_dim=2,
    out_dim=1,
    hidden_size=20,
    n_hidden=5,
    activation=torch.nn.ReLU(),
    bias=True,
    bn=False,
    use_gpu=True,
):
    """
    Initializes the deep neural net model and send to gpu

    Parameters
    ----------
    in_dim: int
        Input dimension.

    out_dim: int
        Output dimension.

    hidden_size: int, default = 10
        Number of nodes in every hidden layer

    n_hidden: int, default = 2
        Number of hidden layers

    activation: ACTIVATION, default = torch.nn.ReLU()
        Activation function to be used by the hidden layers

    bias: bool, default = False
        A boolean indicating if a bias shall be added

    bn: bool, default = False
        A boolean indicating if batch norm shall be applied

    use_gpu: bool, default = True
        A boolean indicating if a gpu is available

    Returns
    -------
    model : Net
        A deep neural net model
    """

    model = Net(
        in_dim,
        out_dim,
        n_hidden=n_hidden,
        hidden_size=hidden_size,
        activation=activation,
        bias=bias,
        bn=bn,
    )

    if use_gpu:
        model = model.cuda()

    return model
