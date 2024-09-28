# imports
import torch
from torch.distributions import Normal
import matplotlib.pyplot as plt
from functools import partial
import sys
from pathlib import Path

# change import path
path = Path.cwd().parents[3]
if str(path) not in sys.path:
    sys.path.append(str(path))

from src.BayesIMPfull import *
from src.kernels import *
from src.GP_utils import *
from src.kernel_utils import *
from src.dgps import *
from src.CBO import *

def main(seed, n=100, n_int=100, niter = 1000, learn_rate = 0.1,
         int_samples=10**5, noise=1.0, front_door = False, int_scale = 4,
         minimise = False):

    """ bayesIMP configs """
    optimise_mu = False
    reg = 1e-3
    quantiles = torch.linspace(0,1,101)[:,None]
    Kernel = GaussianKernel
    force_PD = True

    """ CBO configs """
    n_iter = 10
    xi = 0.0
    update_hyperparameters = False
    noise_init = -10.0
    cbo_reg = 1e-3
        
    """ Setting method """
    if front_door:
        method = "ATT_frontdoor_doB_b"
    else:
        method = "CATE_backdoor_doD_bfixed"

    """ Drawing data """
    A,B,C,D,E,Y,vals,EYdoX = Simulation(n,n_int, 
               mc_samples_EYdoX = int_samples, 
               seed = seed, 
               draw_EYdoX = True, 
               noise = noise,
               method = method, 
               int_min=-int_scale*noise, 
               int_max = int_scale*noise
                )

    """ Variable definitions """
    A = B
    V = C
    if front_door:
        W = B
        doA = vals
        doW = torch.zeros((1,1))     
    else:
        W = D
        doA = torch.zeros((1,1))
        doW = vals
        
    """ Instantiating + training model """
    model = BayesIMP(Kernel_A = GaussianKernel, 
                       Kernel_V = GaussianKernel, 
                       Kernel_W = GaussianKernel,
                       dim_A = A.size()[1], 
                       dim_V = V.size()[1], 
                       dim_W = W.size()[1],
                       samples = 10**5,
                       scale_V_init = Y.var()**0.5/2,
                       noise_Y_init = torch.log(Y.var()/4)
                      )
        
    model.train(Y, A, V, W, niter, learn_rate)

    """ Getting posterior moments """
    mu = model.post_mean(Y, A, V, W=W, doA = doA, doW = doW, reg = reg).detach()
    var = model.post_var(Y, A, V, W=W, doA = doA, doW = doW, reg = reg, diag = True).detach()

    """ Compute out of sample metrics """
    z_quantiles = Normal(0, 1).icdf(1-(1-quantiles)/2)
    posterior_fraction = ((EYdoX - mu).abs() <= var**0.5 @ z_quantiles.T).float()
    rmse = ((EYdoX - mu)**2).mean()**0.5

    """ Get posterior funcs and CBO prior kernel """
    if front_door:
        def mean(X):
            doA = X.reshape(len(X),1)
            doW = torch.zeros((1,1))      
            return model.post_mean(Y,A,V,doA, W=W, doW = doW)
    
        def cov(X, Z, diag = False):
        
            doA = X.reshape(len(X),1)
            doA2 = Z.reshape(len(Z),1)
            doW = torch.zeros((1,1))
            doW2 = torch.zeros((1,1))
        
            return model.post_var(Y,A,V,doA, doA2 = doA2,
                                  W = W, doW = doW, doW2 = doW2,
                                  diag = diag)
    else:
        def mean(X):
            doA = torch.zeros((1,1))
            doW = X.reshape(len(X),1)
            return model.post_mean(Y,A,V,doA, W=W, doW = doW) 
    
        
        def cov(X, Z, diag = False):
        
            doA = torch.zeros((1,1))
            doA2 = torch.zeros((1,1))
            doW = X.reshape(len(X),1)
            doW2 = Z.reshape(len(Z),1)
        
            return model.post_var(Y,A,V,doA, doA2 = doA2,
                                  W = W, doW = doW, doW2 = doW2,
                                  diag = diag)
    
    cbo_kernel = CBOPriorKernel(cov)  

    """ Run CBO """
    # Define a grid of intervention points and precompute E[Y|do(x)]
    doX = vals
    
    # Random search for first intervention point
    torch.manual_seed(seed)
    start = torch.randint(0,99,(1,))[0]
    doXtrain, EYdoXtrain = doX[start].reshape(1,1), EYdoX[start].reshape(1,1)
    
    # Run CBO iters
    doXeval, EYdoXeval = causal_bayesian_optimization(X_train = doXtrain, 
                                                        y_train = EYdoXtrain, 
                                                        kernel = cbo_kernel, 
                                                        mean = mean,
                                                        X_test = doX, 
                                                        Y_test = EYdoX, 
                                                        n_iter = n_iter, 
                                                        update_hyperparameters = update_hyperparameters,
                                                        xi = xi, 
                                                        print_ = False, 
                                                        minimise = minimise,
                                                        noise_init = noise_init,
                                                        reg = cbo_reg)
    
    """ Returning outputs """
    if front_door:
        obs_data = [A,Y]
        int_data = [doA,EYdoX]
    else:
        obs_data = [W,Y]
        int_data = [doW,EYdoX]   
    return {"name" : "bayesimp_frontdoor={0}_minimise={1}".format(front_door,minimise),
                "rmse" : rmse, 
                "cal_levels" : quantiles,
                "post_levels" : posterior_fraction,
                "post_moments" : [mu,var],
                "obs_data" : obs_data,
                "int_data" : int_data,
                "doXeval" : doXeval,
                "EYdoXeval" : EYdoXeval
               }