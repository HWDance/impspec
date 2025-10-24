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
         int_samples=10**5, noise=1.0, front_door = False, int_scale = 1,
         minimise = False, add_base_kernel_BO = False):

    """ bayesIMP configs """
    optimise_mu = False
    reg = 1e-3
    quantiles = torch.linspace(0,1,101)
    Kernel = GaussianKernel
    force_PD = True
        
    """ Setting method """
    method = "CATE_backdoor_doD_bfixed"

    """ Drawing data """
    A,B,C,D,E,Y,vals,EYdoX = Simulation(n,n_int, 
               mc_samples_EYdoX = int_samples, 
               seed = seed, 
               draw_EYdoX = True, 
               noise = noise,
               method = method, 
               int_min=-int_scale*noise, 
               int_max = int_scale*noise,
               discrete_D = True,
                )

    """ Variable definitions """
    A = B
    V = C
    W = D.float()
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
    cov = model.post_var(Y, A, V, W=W, doA = doA, doW = doW, reg = reg, diag = False).detach()

    mu = mu[1] - mu[0]
    var = cov[1,1] + cov[0,0] -2 * cov[1,0]

    """ Compute out of sample metrics """
    z_quantiles = Normal(0, 1).icdf(1-(1-quantiles)/2)
    posterior_fraction = (mu.abs() <= var**0.5 * z_quantiles).float()
    rmse = mu.abs()
    
    """ Returning outputs """
    obs_data = [W,Y]
    int_data = [doW,EYdoX]   
    return {"name" : "bayesimp_frontdoor={0}_minimise={1}".format(front_door,minimise),
                "rmse" : rmse, 
                "cal_levels" : quantiles,
                "post_levels" : posterior_fraction,
                "post_moments" : [mu,var],
                "obs_data" : obs_data,
                "int_data" : int_data,
                "doXeval" : [],
                "EYdoXeval" : []
               }