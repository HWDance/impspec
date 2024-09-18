import torch
from torch.distributions import MultivariateNormal, Normal

def GPML(Y, X, kernel_X, noise_Y,reg = 1e-4, force_PD = False):
    n = len(X)
    K_xx = kernel_X.get_gram(X,X)
    if force_PD:
        K_xx = (K_xx+K_xx.T)/2
    return MultivariateNormal(torch.zeros(n),K_xx+(noise_Y+reg)*torch.eye(n)).log_prob(Y).sum()

def GPfeatureML(Y, X, kernel_Y, kernel_X, noise_feat, reg = 1e-4):
    n = len(Y)
    R_yy = kernel_Y.get_gram(Y, Y)
    K_yy = kernel_Y.get_gram_base(Y, Y)
    K_xx = kernel_X.get_gram(X, X)
    K_x = K_xx + (noise_feat+reg)*torch.eye(n)
    R_y = R_yy+reg*torch.eye(n)
    ml =  -(n/2*(torch.logdet(K_x)+torch.logdet(R_y))
             +1/2*(torch.trace(torch.linalg.solve(K_x, K_yy @ torch.linalg.solve(R_y, K_yy))))
            )
    return ml

def GPmercerML(Y, X, kernel_Y, kernel_X, noise_feat,reg = 1e-4, force_PD = False):
    n = len(Y)
    K_yy = kernel_Y.get_gram_base(Y, Y)
    K_xx = kernel_X.get_gram(X, X)
    K_x = K_xx + (noise_feat+reg)*torch.eye(n)
    if force_PD:
        K_x = 0.5*(K_x + K_x.T)
    ml =  -(K_yy[0,0]*1/2*torch.logdet(K_x)
             +1/2*torch.trace(torch.linalg.solve(K_x, K_yy))
            )
    return ml

def GPpostmean(Y, X, Xtest, kernel, noise, reg = 1e-4):
    n,m = len(X),len(Xtest)
    K_xx = kernel.get_gram(X,X)
    K_xxtest = kernel.get_gram(X,Xtest)
    K_x = K_xx + (noise+reg)*torch.eye(n)
    return  K_xxtest.T @ torch.linalg.solve(K_x, Y)

def GPpostvar(X, Xtest, kernel, noise, reg = 1e-4, latent = False):
    n,m = len(X),len(Xtest)
    K_xx = kernel.get_gram(X,X)
    K_xxtest = kernel.get_gram(X,Xtest)
    K_xtestxtest= kernel.get_gram(Xtest,Xtest)
    K_x = K_xx + (noise+reg)*torch.eye(n)
    return K_xtestxtest - K_xxtest.T @ torch.linalg.solve(K_x, K_xxtest)+(noise*torch.eye(m))*(not latent)
    
def GP_cal(Y, mean, var, levels):
    """
    Y:
        N x 1 array of outputs
    mean:
        N x 1 array of means
    var:
        N x 1 array of variances
    levels:
        P x 1 array of confidence levels
    """
    
    z_quantiles = Normal(0, 1).icdf(1-(1-levels)/2)
    posterior_fraction = ((Y - mean).abs() <= var**0.5 @ z_quantiles.T).float().mean(0)

    return posterior_fraction

    