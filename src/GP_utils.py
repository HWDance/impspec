import torch
from torch.distributions import MultivariateNormal

def GPML(Y, X, kernel_X, noise_Y):
    n = len(Y)
    K_xx = kernel_X.get_gram(X,X)
    return MultivariateNormal(torch.zeros(n),K_xx+noise_Y*torch.eye(n)).log_prob(Y)

def GPfeatureML(Y, X, kernel_Y, kernel_X, noise_feat, reg = 1e-4):
    n = len(Y)
    R_yy = kernel_Y.get_gram(Y, Y)
    K_yy = kernel_Y.get_gram_base(Y, Y)
    K_xx = kernel_X.get_gram(X, X)
    K_x = K_xx + noise_feat*torch.eye(n)
    R_y = R_yy+reg*torch.eye(n)
    ml =  -(+n/2*(torch.logdet(K_x)+torch.logdet(R_y))
             +1/2*(torch.trace(torch.linalg.solve(K_x, K_yy @ torch.linalg.solve(R_y, K_yy))))
            )
    return ml

def GPpostvar(X, Xtest, kernel_params, noise, latent = False):
    n,m = len(X),len(Xtest)
    K_xx = get_gram_gaussian(X,X,kernel_params)
    K_xxtest = get_gram_gaussian(X,Xtest,kernel_params)
    K_xtestxtest= get_gram_gaussian(Xtest,Xtest,kernel_params)
    
    return K_xtestxtest - K_xxtest.T @ torch.linalg.solve(K_xx + noise*torch.eye(n), K_xxtest)+(noise*torch.eye(m))*(not latent)