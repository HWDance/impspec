import torch

def f_x(Z,coefs):
    return torch.sin(Z*coefs)

def f_y(X,coefs):
    return (torch.sin(X)*coefs).T.sum(0)