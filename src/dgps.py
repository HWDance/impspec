import torch
from torch.distributions import Normal
from math import pi    

def f_x(Z,coefs):
    return torch.sin(Z*coefs)

def f_y(X,coefs):
    return (torch.sin(X)*coefs).T.sum(0)

def BayesIMP_Abelation(n,ntest, sigma_x,sigma_y,sigma_t, get_ET = True, mc_samples = 10**3):
    X = Normal(0,sigma_x).sample((n+ntest,1))
    Y = X*torch.cos(pi*X) + Normal(0,sigma_y).sample((n+ntest,1))
    T = 0.5*Y*torch.cos(Y) + Normal(0,sigma_t).sample((n+ntest,1))

    Xtrain, Ytrain, Ttrain = X[:n], Y[:n], T[:n]
    Xtest, Ytest, Ttest =  X[n:], Y[n:], T[n:]

    if get_ET:
        Ysample = (Xtest*torch.cos(pi*Xtest)) @ torch.ones((1,mc_samples)) + Normal(0,sigma_y).sample((ntest,mc_samples))
        ET = (0.5*Ysample*torch.cos(Ysample)).mean(1) + Normal(0,sigma_t).sample((ntest,mc_samples)).mean(1)
        
    return Xtrain, Ytrain, Ttrain, Xtest, Ytest, Ttest, ET
    