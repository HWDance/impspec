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

from src.baselineGPfull import *
from src.causalKLGPfull import *
from src.kernels import *
from src.GP_utils import *
from src.kernel_utils import *
from src.dgps import *
from src.CBO import *

def main(seed, n=100, n_int=100, niter = 1000, learn_rate = 0.1,
         int_samples=10**5, noise=1.0, front_door = False, int_scale = 4,
        minimise = False):

    """ model configs """
    quantiles = torch.linspace(0,1,101)
    reg = 1e-3
    Kernel = GaussianKernel
    force_PD = True
    single_kernel = False
    error_samples = 10**2
    gp_samples = 10**3

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
    model = baselineGP(Kernel_A = GaussianKernel, 
                       Kernel_V = GaussianKernel, 
                       Kernel_W = GaussianKernel,
                       dim_A = A.size()[1], 
                       dim_V = V.size()[1], 
                       dim_W = W.size()[1],
                       scale_V_init = Y.var()**0.5/2,
                       noise_Y_init = torch.log(Y.var()/4),
                       single_kernel = single_kernel)
        
    model.train(Y, A, V, W, niter, learn_rate, force_PD = True)

    """ Getting posterior samples """
    EYdoX_samples, _ = model.marginal_post_sample(Y,A,V,doA, W = W, doW = doW, reg = 1e-3, 
                     error_samples = error_samples, gp_samples = gp_samples)

    """ Compute out of sample metrics """
    upper_quantiles = 1-(1-quantiles)/2
    lower_quantiles = (1-quantiles)/2
    u = (upper_quantiles*(gp_samples-1)).int()
    l = (lower_quantiles*(gp_samples-1)).int()
    EY_u = EYdoX_samples.sort(1)[0][:,u]
    EY_l = EYdoX_samples.sort(1)[0][:,l]

    posterior_fraction = ((EY_u>=EYdoX)*(EY_l<=EYdoX)).float()   
    rmse = ((EYdoX - EYdoX_samples.mean(1).reshape(n_int,1))**2).mean()**0.5

    """ Get posterior funcs and CBO prior kernel """

    """ Initialise cbo model """
    model_EY = causalKLGP(Kernel_A = GaussianKernel, 
                       Kernel_V = GaussianKernel, 
                       Kernel_W = GaussianKernel,
                       dim_A = A.size()[1], 
                       dim_V = V.size()[1], 
                       dim_W = W.size()[1],
                       samples = 10**5,
                       scale_V_init = (Y**2).var()**0.5/2,
                       noise_Y_init = torch.log(Y.var()/4)
                      )
    
    """ Train cbo model """
    model_EY.train(Y,A,V,W,niter = niter,learn_rate = learn_rate, reg = reg, force_PD = True)    

    """ Get posterior funcs and CBO prior kernel """
    if front_door:
        def mean(X):
            doA = X.reshape(len(X),1)
            doW = torch.zeros((1,1))      
            return model_EY.post_mean(Y,A,V,doA, W=W, doW = doW,
                                      reg = reg)
    
        def var(X):
        
            doA = X.reshape(len(X),1)
            doW = torch.zeros((1,1))
            EYdoX = mean(X)
            EY2doX = model_EY.post_mean(Y**2,A,V,doA, W=W, doW = doW,
                                      reg = reg)
            return (EY2doX - EYdoX**2).abs()
            
    else:
        def mean(X):
            doW = X.reshape(len(X),1)
            doA = torch.zeros((1,1))      
            return model_EY.post_mean(Y,A,V,doA, W=W, doW = doW,
                                      reg = reg)
    
        def var(X):
        
            doW = X.reshape(len(X),1)
            doA = torch.zeros((1,1))
            EYdoX = mean(X)
            EY2doX = model_EY.post_mean(Y**2,A,V,doA, W=W, doW = doW,
                                      reg = reg)
            return (EY2doX - EYdoX**2).abs()

    if front_door:
        medheur = median_heuristic(A)
    else:
        medheur = median_heuristic(W)
        
    rbf_kernel = GaussianKernel(lengthscale=torch.tensor([medheur]).requires_grad_(True), 
                                scale=torch.tensor(Y.var()**0.5/2).requires_grad_(True))
    cbo_kernel = CausalKernel(
            estimate_var_func=var,
            base_kernel=rbf_kernel,
            add_base_kernel=True
        ) 

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
    return {"name" : "baseline_frontdoor={0}_minimise={1}".format(front_door, minimise),
                "rmse" : rmse, 
                "cal_levels" : quantiles,
                "post_levels" : posterior_fraction,
                "post_samples" : [EYdoX_samples],
                "obs_data" : obs_data,
                "int_data" : int_data,
                "doXeval" : doXeval,
                "EYdoXeval" : EYdoXeval
               }