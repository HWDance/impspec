# imports
import torch
import sys
from pathlib import Path

path = Path.cwd().parents[3]
if str(path) not in sys.path:
    sys.path.append(str(path))

from src.BayesIMP import *
from src.kernels import *
from src.dgps import *

# main
def main(seed, n,ntest,d,noise, niter = 500, learn_rate = 0.1, optimise_mu = True, exact = True, mc_samples = 100):
    
    """ Fixed configs """
    default_nu = 1.0
    nulist = 2**torch.linspace(-5,5,11)
    cal_levels = torch.tensor([0.1,0.3,0.5,0.7,0.9])
    calibrate_norm = 1
    reg = 1e-4
    Kernel = GaussianKernel
    
    """ Draw data """
    Z, V, Y, doZ, YdoZ, EYdoZ = Abelation(n, ntest, d, noise, doZlower = 0, doZupper = 1, mc_samples_EYdoZ = 10**4, seed = seed) 
    
    """ Initialise model """
    model = BayesIMP(Kernel_A = Kernel, 
                   Kernel_V = Kernel, 
                   Kernel_Z = [],
                   dim_A = Z.size()[1], 
                   dim_V = V.size()[1], 
                   samples = 10**5,
                   exact = exact)

    """ Train model """
    model.train(Y,Z,V,niter,learn_rate, optimise_measure = optimise_mu, mc_samples = mc_samples)
    
    """ Get Posterior moments """
    mean = model.post_mean(Y, Z, V, doZ).detach()
    var = model.post_var(Y, Z, V, doZ, reg = reg, latent = True).detach()
    var_noise = model.post_var(Y, Z, V, doZ, reg = reg, latent = False).detach()


    """ Compute out of sample metrics """
    levels = torch.linspace(0,1,21)[:,None]
    z_quantiles = Normal(0, 1).icdf(1-(1-levels)/2)
    posterior_fraction_f = ((EYdoZ - mean).abs() <= var**0.5 @ z_quantiles.T).float().mean(0)
    posterior_fraction_y = ((YdoZ - mean).abs() <= var_noise**0.5 @ z_quantiles.T).float().mean(0)
    rmse = ((EYdoZ - mean)**2).mean()**0.5

    return {"name" : "bayesimp",
            "rmse" : rmse, 
           "cal_levels" : levels,
           "post_levels_f" : posterior_fraction_f,
           "post_levels_y" : posterior_fraction_y
           }