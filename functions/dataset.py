
import numpy as np
import torch
from torch import nn

import copy
import random

import os
## Distributions 

def generate_gaussian_parity(n, cov_scale=1, angle_params=None, k=1, acorn=None):
    """ Generate Gaussian XOR, a mixture of four Gaussians elonging to two classes. 
    Class 0 consists of negative samples drawn from two Gaussians with means (−1,−1) and (1,1)
    Class 1 comprises positive samples drawn from the other Gaussians with means (1,−1) and (−1,1) 
    """
#     means = [[-1.5, -1.5], [1.5, 1.5], [1.5, -1.5], [-1.5, 1.5]]
    means = [[-1, -1], [1, 1], [1, -1], [-1, 1]]
    blob = np.concatenate(
        [
            np.random.multivariate_normal(
                mean, cov_scale * np.eye(len(mean)), size=int(n / 4)
            )
            for mean in means
        ]
    )

    X = np.zeros_like(blob)
    Y = np.concatenate([np.ones((int(n / 4))) * int(i < 2) for i in range(len(means))])
    X[:, 0] = blob[:, 0] * np.cos(angle_params * np.pi / 180) + blob[:, 1] * np.sin(
        angle_params * np.pi / 180
    )
    X[:, 1] = -blob[:, 0] * np.sin(angle_params * np.pi / 180) + blob[:, 1] * np.cos(
        angle_params * np.pi / 180
    )
    return X, Y.astype(int)
        

def get_dataset(N=1000, one_hot=False, cov_scale=1, include_hybrid=False):
    """
     Generate the Gaussian XOR dataset and move to gpu
    """
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(0)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        
    if include_hybrid:
        D_x, D_y = generate_gaussian_parity(cov_scale=cov_scale, n=2*N, angle_params=0)
        D_perm = np.random.permutation(2*N)
        D_x, D_y  = D_x[D_perm,:], D_y[D_perm]
        train_x, train_y = D_x[:N], D_y[:N]
        ghost_x, ghost_y = D_x[N:], D_y[N:]
        hybrid_sets = []
        rand_idx = random.sample(range(0,N-1), N//10)
        for rand_i in rand_idx:
            hybrid_x, hybrid_y = np.copy(train_x), np.copy(train_y)
            hybrid_x[rand_i], hybrid_y[rand_i] = ghost_x[rand_i], ghost_y[rand_i]
            hybrid_x = torch.FloatTensor(hybrid_x)
            hybrid_y = (torch.FloatTensor(hybrid_y).unsqueeze(-1))
            hybrid_x, hybrid_y = hybrid_x.cuda(), hybrid_y.cuda()
            hybrid_sets.append((hybrid_x, hybrid_y))
    else:
        train_x, train_y = generate_gaussian_parity(cov_scale=cov_scale, n=N, angle_params=0)
        train_perm = np.random.permutation(N)
        train_x, train_y = train_x[train_perm,:], train_y[train_perm] 
    test_x, test_y = generate_gaussian_parity(cov_scale=cov_scale, n=2*N, angle_params=0)
    
    test_perm = np.random.permutation(2*N)
    test_x, test_y  = test_x[test_perm,:], test_y[test_perm]
    
    train_x = torch.FloatTensor(train_x)
    test_x = torch.FloatTensor(test_x)

    train_y = (torch.FloatTensor(train_y).unsqueeze(-1))#[:,0]
    test_y = (torch.FloatTensor(test_y).unsqueeze(-1))#[:,0]
    
    if one_hot:
        train_y = torch.nn.functional.one_hot(train_y[:,0].to(torch.long))
        test_y = torch.nn.functional.one_hot(test_y[:,0].to(torch.long))
    
    # move to gpu
    if use_cuda:
        train_x, train_y = train_x.cuda(), train_y.cuda()
        test_x, test_y = test_x.cuda(), test_y.cuda()
        
    if include_hybrid:
        return train_x, train_y, test_x, test_y, hybrid_sets
    
    return train_x, train_y, test_x, test_y
