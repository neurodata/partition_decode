import numpy as np
import torch
from torch import nn

import copy
import random

import os

## Distributions


def generate_gaussian_parity(
    n_samples,
    means=None,
    cov_scale=1,
    angle_params=None,
    random_state=None,
):
    """
    Generate 2-dimensional Gaussian XOR distribution, a mixture of four Gaussian belonging to two classes.
    (Classic XOR problem but each point is the center of a Gaussian blob distribution)

    Parameters
    ----------
    n_samples : int
        Total number of points divided among the four clusters with equal probability.

    means : ndarray of shape [n_centers,2], default=None
        The coordinates of the center of total n_centers blobs.

    cov_scale : float, default=1
        The standard deviation of the blobs.

    angle_params: float, default=None
        Number of radians to rotate the distribution by.

    random_state : int, RandomState instance, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.

    y : array of shape [n_samples]
        The integer labels for cluster membership of each sample.
    """

    if random_state != None:
        np.random.seed(random_state)

    if means is None:
        means = [[-1, -1], [1, 1], [1, -1], [-1, 1]]

    if angle_params is None:
        angle_params = np.random.uniform(0, 2 * np.pi)

    blob = np.concatenate(
        [
            np.random.multivariate_normal(
                mean, cov_scale * np.eye(len(mean)), size=int(n_samples / 4)
            )
            for mean in means
        ]
    )

    X = np.zeros_like(blob)
    Y = np.concatenate(
        [np.ones((int(n_samples / 4))) * int(i < 2) for i in range(len(means))]
    )
    X[:, 0] = blob[:, 0] * np.cos(angle_params * np.pi / 180) + blob[:, 1] * np.sin(
        angle_params * np.pi / 180
    )
    X[:, 1] = -blob[:, 0] * np.sin(angle_params * np.pi / 180) + blob[:, 1] * np.cos(
        angle_params * np.pi / 180
    )

    return X, Y.astype(int)


def get_dataset(
    n_samples=1000, one_hot=False, cov_scale=1, include_hybrid=False, random_state=None
):
    """
    Generate the Gaussian XOR dataset and move it to gpu

    Parameters
    ----------
    n_samples : int
        Total number of points in the Gaussian XOR dataset.

    one_hot : bool, default=False
        A boolean indicating if the label should one hot encoded.

    cov_scale : float, default=1
        The standard deviation of the blobs.

    include_hybrid: bool, default=False
        A boolean indicating if hybrid set should be included for computing average stability.

    random_state : int, RandomState instance, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    train_x : Tensor [n_samples, 2]
        Training set features

    train_y: Tensor [n_samples]
        Training set labels

    test_x : Tensor [n_samples, 2]
        Test set features

    test_y: Tensor [n_samples]
        Test set labels
    """

    use_gpa = torch.cuda.is_available()
    if use_gpa:
        torch.cuda.set_device(0)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    if include_hybrid:
        D_x, D_y = generate_gaussian_parity(
            cov_scale=cov_scale,
            n_samples=(2 * n_samples),
            angle_params=0,
            random_state=random_state,
        )
        D_perm = np.random.permutation(2 * n_samples)
        D_x, D_y = D_x[D_perm, :], D_y[D_perm]
        train_x, train_y = D_x[:n_samples], D_y[:n_samples]
        ghost_x, ghost_y = D_x[n_samples:], D_y[n_samples:]
        hybrid_sets = []
        rand_idx = random.sample(range(0, n_samples - 1), n_samples // 10)
        for rand_i in rand_idx:
            hybrid_x, hybrid_y = np.copy(train_x), np.copy(train_y)
            hybrid_x[rand_i], hybrid_y[rand_i] = ghost_x[rand_i], ghost_y[rand_i]
            hybrid_x = torch.FloatTensor(hybrid_x)
            hybrid_y = torch.FloatTensor(hybrid_y).unsqueeze(-1)
            if use_gpa:
                hybrid_x, hybrid_y = hybrid_x.cuda(), hybrid_y.cuda()
            hybrid_sets.append((hybrid_x, hybrid_y))
    else:
        train_x, train_y = generate_gaussian_parity(
            cov_scale=cov_scale,
            n_samples=n_samples,
            angle_params=0,
            random_state=random_state,
        )
        train_perm = np.random.permutation(n_samples)
        train_x, train_y = train_x[train_perm, :], train_y[train_perm]

    test_x, test_y = generate_gaussian_parity(
        cov_scale=cov_scale,
        n_samples=2 * n_samples,
        angle_params=0,
        random_state=random_state,
    )

    test_perm = np.random.permutation(2 * n_samples)
    test_x, test_y = test_x[test_perm, :], test_y[test_perm]

    train_x = torch.FloatTensor(train_x)
    test_x = torch.FloatTensor(test_x)

    train_y = torch.FloatTensor(train_y).unsqueeze(-1)  # [:,0]
    test_y = torch.FloatTensor(test_y).unsqueeze(-1)  # [:,0]

    if one_hot:
        train_y = torch.nn.functional.one_hot(train_y[:, 0].to(torch.long))
        test_y = torch.nn.functional.one_hot(test_y[:, 0].to(torch.long))

    # move to gpu
    if use_gpa:
        train_x, train_y = train_x.cuda(), train_y.cuda()
        test_x, test_y = test_x.cuda(), test_y.cuda()

    if include_hybrid:
        return train_x, train_y, test_x, test_y, hybrid_sets

    return train_x, train_y, test_x, test_y
