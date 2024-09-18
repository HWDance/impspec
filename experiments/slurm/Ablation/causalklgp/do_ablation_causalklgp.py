# imports
import torch
import sys
from pathlib import Path

path = Path.cwd().parents[3]
if str(path) not in sys.path:
    sys.path.append(str(path))

from src.causalKLGP import *
from src.kernels import *
from src.dgps import *

# main
def main(seed, n,ntest,d,noise, niter = 500, learn_rate = 0.1, calibrate = True, sample_split = False,
         marginal_loss = False, retrain_hypers = False):
    
    """ Fixed configs """
    default_nu = 1.0
    cal_nulist = 2**torch.linspace(-4,4,5)
    quantiles = torch.linspace(0,1,101)[:,None]
    reg = 1e-4
    Kernel = GaussianKernel
    
    """ Draw data """
    Z, V, Y, doZ, YdoZ, EYdoZ = Abelation(n, ntest, d, noise, doZlower = 0, doZupper = 1, mc_samples_EYdoZ = 10**4, seed = seed) 
    
    """ Initialise model """
    model = causalKLGP(Kernel_A = Kernel, 
                   Kernel_V = Kernel, 
                   Kernel_Z = [],
                   dim_A = Z.size()[1], 
                   dim_V = V.size()[1], 
                   samples = 10**5)
    
    """ Train + Calibrate model """
    if calibrate:
        Post_levels, Calibration_losses = model.frequentist_calibrate(Y, V, Z, doZ,
                                                                     nulist = cal_nulist,
                                                                     sample_split = sample_split,
                                                                     marginal_loss = marginal_loss,
                                                                     retrain_hypers = retrain_hypers
                                                                    )
        best_ind = torch.where(Calibration_losses == Calibration_losses.min())[0][0]
        nu_best = cal_nulist[best_ind]
    else:
        nu_best = default_nu
    if (not calibrate) or sample_split: 
        model.train(Y, Z, V, niter,learn_rate)
    
    """ Get Posterior moments """
    mean = model.post_mean(Y, Z, V, doZ).detach()
    var = model.post_var(Y, Z, V, doZ, reg = reg, latent = True, nu = nu_best).detach()
    var_noise = model.post_var(Y, Z, V, doZ, reg = reg, latent = False, nu = nu_best).detach()


    """ Compute out of sample metrics """
    z_quantiles = Normal(0, 1).icdf(1-(1-quantiles)/2)
    posterior_fraction = ((EYdoZ - mean).abs() <= var**0.5 @ z_quantiles.T).float()
    rmse = ((EYdoZ - mean)**2).mean()**0.5

    return {"name" : "causalklgp_cal={0}_split={1}".format(calibrate, sample_split),
            "rmse" : rmse, 
           "cal_levels" : quantiles,
           "post_levels" : posterior_fraction
           }