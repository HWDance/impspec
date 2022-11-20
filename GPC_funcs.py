# imports
import torch
import gpytorch
import matplotlib.pyplot as plt
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from functools import partial

def f_x(Z,coefs):
    return torch.sin(Z*coefs)

def f_y(X,coefs):
    return (torch.sin(X)*coefs).T.sum(0)

def get_gram_gaussian(X1,X2,kernel_params):
    """
    X1: nxd matrix of inputs (d dimensions)
    X2: mxd matrix of inputs (d dimensions)
    kernel_params: (d+1)x1 vector of scale and lengthscale parameters (scale first)
    
    returns: nxm gram matrix K_xx
    """
    
    n,m = len(X1),len(X2)
    K_xx = torch.exp(-0.5*torch.cdist(X1/kernel_params[1:], X2/kernel_params[1:], p=2.0)**2)
    return K_xx*kernel_params[0]**2

def get_gram_gaussian_1d(X1,X2,kernel_params):
    """
    X1: nxd matrix of inputs (d dimensions)
    X2: mxd matrix of inputs (d dimensions)
    kernel_params: 2x1 vector of scale and lengthscale parameters (scale first)
    
    returns: nxm gram matrix K_xx
    """
    
    n,m = len(X1),len(X2)
    K_xx = torch.exp(-0.5/kernel_params[1:]**2*(X1-X2.T)**2)
    return K_xx*kernel_params[0]**2

def GP_post_var(X, Xtest, kernel_params, noise,latent = False):
    n,m = len(X),len(Xtest)
    K_xx = get_gram_gaussian(X,X,kernel_params)
    K_xxtest = get_gram_gaussian(X,Xtest,kernel_params)
    K_xtestxtest= get_gram_gaussian(Xtest,Xtest,kernel_params)
    
    return K_xtestxtest - K_xxtest.T @ torch.linalg.solve(K_xx + noise*torch.eye(n), K_xxtest)+(noise*torch.eye(m))*(not latent)

def kernel_approximations(X_train, kernel_params, mc_approx = True, mc_samples = 10**3, dist = "normal"):
    
    if mc_approx:
        if dist == "unif":
            measure = torch.distributions.uniform.Uniform(X_train.min(),X_train.max())
        else:
            measure = torch.distributions.normal.Normal(0,1)            
        
        X_sample = measure.sample((mc_samples,len(X_train.T)))
        K_s = get_gram_gaussian(X_train, X_sample, kernel_params)
        K_ss = get_gram_gaussian(X_sample, X_sample, kernel_params)
        K_xx_tilde = K_s @ K_s.T/mc_samples
        # Get eigenvalues
        eigs = torch.eig(K_ss)[0]/mc_samples
        
    return K_xx_tilde, K_ss, eigs

def get_smoothed_Gaussian_kernel(X_train, kernel_params, mc_approx = True, mc_samples = 10**3, nu = 1):
    
    if mc_approx:
        measure = torch.distributions.normal.Normal(0,1)
        X_sample = nu*measure.sample((mc_samples,len(X_train.T)))
        K_s = get_gram_gaussian(X_train, X_sample, kernel_params)
        K_xx_tilde = K_s @ K_s.T/mc_samples
    
    else: 
        theta = kernel_params[1]**2
        K_xx_tilde = kernel_params[0]**4*(theta/(2+theta))**0.5*torch.exp(-1/(2*theta)*(X_train**2+X_train.T**2)+1/2*(X_train+X_train.T)**2*(1/(theta*(theta+2))))
    return K_xx_tilde

def get_nuclear_kernel(X1,X2, kernel_params,nu=1): # for N(0,nu^2) measure
    
    d = len(X1.T)
    M = torch.diag(1/kernel_params[1:]**2)
    A = 2*M+torch.eye(d)/nu**2
    B_inv = 1/(1/(2*M)+torch.eye(d)*nu**2)
    normaliser = torch.diag(A).prod()**-0.5
    exponent1 = -1/2*torch.cdist(X1/(kernel_params[1:]*2**0.5), X2/(kernel_params[1:]*2**0.5), p=2.0)**2
    exponent2 = -1/8*torch.cdist(X1*torch.diag(B_inv)**0.5, -X2*torch.diag(B_inv)**0.5, p=2.0)**2
    
    return kernel_params[0]**4*normaliser*torch.exp(exponent1+exponent2)*nu**-d

def get_Gaussian_kernel_eigs(X_train,kernel_params, mc_approx = True, mc_samples = 10**3, nu = 1): # 1d case only for P(x) = N(0,1)
    
    if mc_approx: 
        measure = torch.distributions.normal.Normal(0,nu)            
        X_sample = measure.sample((mc_samples,len(X_train.T)))
        K_sample = get_gram_gaussian(X_sample, X_sample, kernel_params)
        eigs = torch.eig(K_sample)[0]/mc_samples
    
    else: 
        a = 1/4
        b = 1/(2*kernel_params[1]**2)
        c = (a**2+2*a*b).sqrt()
        A = a + b + c
        B = b/A
        eigs = kernel_params[0]**2*(2*a/A).sqrt()*B**(torch.linspace(0,mc_samples-1,mc_samples))
    return eigs

def marginal_posterior_mean(Y_train, X_train, Z_train, z_test, 
                            kernel_params_x, kernel_params_z, 
                            y_noise, z_noise, reg=1e-8):
    """
    vectorised for multiple (W*,Z*), fixed A* = [a*,...,a*]^T
    returns: E[Y|a,w,z,D]
    """
    n = len(Y_train)
    K_xx = get_gram_gaussian(X_train, X_train, kernel_params_x)
    K_zz,k_z = get_gram_gaussian(Z_train, Z_train, kernel_params_z), get_gram_gaussian(z_test, Z_train, kernel_params_z)
    I = torch.eye(n)
    
    alpha_y = torch.linalg.solve(K_xx + (y_noise+reg)*I,Y_train)
    alpha_phi = torch.linalg.solve(K_zz + (z_noise+reg)*I, K_xx @ alpha_y)
    
    return k_z @ alpha_phi

def marginal_posterior_covariance(Y_train,X_train,Z_train,
                                       z1,z2, noise_y, noise_phi,
                                       kernel_params_x,kernel_params_z,K_xx_tilde, eigsum, latent = False):
    n = len(X_train)
    
    # get posterior variance of phi(x)
    kernel_params_phi = kernel_params_z
    if z2!=[]:
        phi_post_cov = GP_post_var(Z_train, torch.cat((z1,z2),0), kernel_params_phi,noise_phi, latent)[1,0]
    else:
        phi_post_cov = GP_post_var(Z_train, z1, kernel_params_phi, noise_phi, latent)
    
     # get gram matrices
    K_xx,K_zz = (get_gram_gaussian(X_train, X_train, kernel_params_x),
                 get_gram_gaussian(Z_train, Z_train, kernel_params_z))
    k_z1 = get_gram_gaussian(z1, Z_train, kernel_params_z)
    if z2!=[]:
        k_z2 = get_gram_gaussian(z2, Z_train, kernel_params_z)
    else:
        k_z2= k_z1
    
    # compute matrix-vector products
    K_phi = K_zz+noise_phi*torch.eye(n)
    K_y = K_xx+noise_y*torch.eye(n)
    alpha_y = torch.linalg.solve(K_y, Y_train)
    B = torch.linalg.solve(K_y, K_xx_tilde)
    alpha1 = torch.linalg.solve(K_phi,K_xx)
    alpha2 = torch.linalg.solve(K_phi,k_z2.T)
    DDKxx =  torch.linalg.solve(K_y, K_xx)
    V1 = phi_post_cov*(alpha_y.T @ K_xx_tilde @ alpha_y).view(1,)
    V2 = phi_post_cov*(eigsum - torch.trace(B))
    V3 = k_z1 @ alpha1 @ (torch.eye(n) - DDKxx) @ alpha2+noise_y*(not latent)*(z2==[])*torch.eye(len(z1))
    return V1+V2+V3

def BayesImp_mean(Y,X,Z,Ztest,kernel_params_r,kernel_params_z,noise_r,noise_z, mc_approx = False, mc_samples = 10**3, nu = 1):
    n = len(Y)
    
    # getting kernel matrices
    K_xx,K_zz,k_ztest = (get_gram_gaussian(X, X, kernel_params_r),
                         get_gram_gaussian(Z, Z, kernel_params_z),
                         get_gram_gaussian(Ztest, Z, kernel_params_z))
    if mc_approx:
        R_xx = get_smoothed_Gaussian_kernel(X, kernel_params_r, mc_approx = True, mc_samples = mc_samples, nu = nu)
    else:
        R_xx = get_nuclear_kernel(X,X, kernel_params_r, nu = nu)
    K_z = K_zz+noise_z*torch.eye(n)
    R_x = R_xx+noise_r*torch.eye(n)
    
    A_z = torch.linalg.solve(K_z,k_ztest.T).T
    alpha_r = torch.linalg.solve(R_x,Y)
    
    return  A_z @ R_xx @ alpha_r

def BayesImp_var(Y,X,Z,Ztest,kernel_params_r,kernel_params_z,noise_r,noise_z, mc_approx = False, mc_samples = 10**3, nu = 1):
    
    n = len(Y)
    
    # getting kernel matrices
    K_xx,K_zz,k_ztest = (get_gram_gaussian(X, X, kernel_params_r),
                         get_gram_gaussian(Z, Z, kernel_params_z),
                         get_gram_gaussian(Ztest, Z, kernel_params_z))
    if mc_approx:
        R_xx = get_smoothed_Gaussian_kernel(X, kernel_params_r, mc_approx = True, mc_samples = mc_samples, nu = nu)
    else:
        R_xx = get_nuclear_kernel(X,X, kernel_params_r, nu = nu)
    kpost_ztest = GP_post_var(Z, Ztest, kernel_params_z, noise_z, latent = True)
    K_z = K_zz+noise_z*torch.eye(n)
    K_x = K_xx+noise_r*torch.eye(n)
    R_x = R_xx+noise_r*torch.eye(n)
    R_xx_bar = R_xx - R_xx @ torch.linalg.solve(R_x,R_xx)
    
    # computing matrix vector products
    alpha_z = torch.linalg.solve(K_z,k_ztest.T)
    alpha_y = torch.linalg.solve(K_x,Y)
    KinvR = torch.linalg.solve(K_xx+torch.eye(n)*1e-3,R_xx)
    
    V1 = alpha_z.T @ R_xx_bar @ alpha_z
    V2 = alpha_y.T @ R_xx @ KinvR @ KinvR @ alpha_y * kpost_ztest 
    V3 = torch.trace(torch.linalg.solve(K_xx+torch.eye(n)*1e-3, R_xx_bar @ KinvR)) * kpost_ztest
    
    return V1+V2+V3

def get_CV_splits(X,folds=[]):
    """
    X: nxd matrix
    folds: # CV folds 
    
    Returns: list of folds x training and validation sets
    """
    
    n = len(X)
    if folds==[]:
        folds = n
    n_per_fold = int(n/folds+1-1e-10) # rounds up except if exact integer
    row_count = torch.linspace(0,n-1,n) 
    train_val_sets = list()
    
    for i in range(folds):
        test_inds = ((row_count>= n_per_fold*i)*(row_count<n_per_fold*(i+1)))>0
        train_inds = test_inds==0
        train_val_sets.append([X[test_inds],X[train_inds]])
    
    return train_val_sets

def get_CV_loss(Y,X,Z, 
                kernel_params_x, kernel_params_z, 
                y_noise, z_noise,func,folds = []):
    
    # Randomising order
    """
    n=len(Y)
    inds = torch.randperm(n)
    Y,X,Z = (Y[inds],
                X[inds],
                Z[inds])
    """
    # getting splits
    Y_splits = get_CV_splits(Y,folds)
    X_splits = get_CV_splits(X,folds)
    Z_splits = get_CV_splits(Z,folds)
    
    loss = 0
    
    for i in range(folds):
        mu_i = func(Y_splits[i][1], X_splits[i][1], Z_splits[i][1], Z_splits[i][0],
                                kernel_params_x, kernel_params_z, 
                                y_noise, z_noise)
        loss += ((Y_splits[i][0]-mu_i)**2).sum()
        
    return loss.sqrt()/len(Y)