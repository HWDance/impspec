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

from src.causalKLGPfull import *
from src.kernels import *
from src.GP_utils import *
from src.kernel_utils import *
from src.dgps import *
from src.CBO import *

def main(seed, n=100, n_int=100, niter = 1000, learn_rate = 0.1,
         int_samples=10**5, noise=1.0, front_door = False, int_scale = 1,
        minimise = False, calibrate = True, sample_split = True, add_base_kernel_BO = False):

    """ causalklgp configs """
    default_nu = 1.0
    cal_nulist = 2**torch.linspace(-4,4,9)
    quantiles = torch.tensor([0.8,0.9,0.95])
    reg = 1e-3
    Kernel = GaussianKernel
    force_PD = True
    bootreps = 20
    scale_var = False
        
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
               int_max = int_scale*noise,
               discrete_D = True,
                )

    """ Variable definitions """
    A = B
    V = C   
    W = D
    doA = torch.zeros((1,1))
    doW = vals
    
    """ Instantiating + training model """
    model = causalKLGP(Kernel_A = GaussianKernel, 
                   Kernel_V = GaussianKernel, 
                   Kernel_W = GaussianKernel,
                   dim_A = A.size()[1], 
                   dim_V = V.size()[1], 
                   dim_W = W.size()[1],
                   samples = 10**5,
                   scale_V_init = Y.var()**0.5/2,
                   noise_Y_init = torch.log(Y.var()/4)
                  )

    if calibrate:
        Post_levels, Calibration_losses = model.frequentist_calibrate(Y, V, A, doA, W = W, doW = doW,
                                                                     niter = niter, 
                                                                     learn_rate = learn_rate,
                                                                     bootstrap_replications = bootreps,
                                                                     nulist = cal_nulist,
                                                                     sample_split = sample_split,
                                                                     marginal_loss = False,
                                                                     retrain_hypers = False,
                                                                     average_doA = False,
                                                                     intervention_indices = None,
                                                                     force_PD = force_PD,
                                                                     reg = reg,
                                                                     scale_var = scale_var,
                                                                    )
        best_ind = torch.where(Calibration_losses == Calibration_losses.min())[0][0]
        nu_best = cal_nulist[best_ind]
    else:
        nu_best = default_nu
        
    if (not calibrate) or sample_split: 
        model.train(Y, A, V, W, niter, learn_rate, force_PD = force_PD, reg = reg)   
        
    """ Getting posterior moments """
    mu = model.post_mean(Y, A, V, W=W, doA = doA, doW = doW, reg = reg).detach()
    if scale_var:
        cov = nu_best*model.post_var(Y, A, V, W=W, doA = doA, doW = doW, reg = reg, nu = 1.0, diag = False).detach()
    else:
        cov = model.post_var(Y, A, V, W=W, doA = doA, doW = doW, reg = reg, nu = nu_best, diag = False).detach()
        

    mu = mu[1] - mu[0]
    var = cov[1,1] + cov[0,0] -2*cov[1,0]

    """ Compute out of sample metrics """
    z_quantiles = Normal(0, 1).icdf(1-(1-quantiles)/2)
    posterior_fraction = (mu.abs() <= var**0.5 * z_quantiles).float()
    rmse = mu.abs()

    """ Returning outputs """
    obs_data = [D,Y]
    int_data = [vals,EYdoX]   
    return {"name" : "causalklgp_cal={0}_split={1}_frontdoor={2}_discrete".format(False, False, front_door,minimise),
                "rmse" : rmse, 
                "cal_levels" : quantiles,
                "post_levels" : posterior_fraction,
                "post_moments" : [mu,var],
                "obs_data" : obs_data,
                "int_data" : int_data,
                "doXeval" : [],
                "EYdoXeval" : []
               }