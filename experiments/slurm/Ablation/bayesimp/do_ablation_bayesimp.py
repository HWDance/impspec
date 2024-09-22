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
def main(seed, n,ntest,d,noise, niter = 500, learn_rate = 0.1, optimise_mu = True, exact = True, mc_samples = 100, kernel = "gaussian"):

    torch.manual_seed(seed)
    
    """ Fixed configs """
    default_nu = 1.0
    reg = 1e-3
    quantiles = torch.linspace(0,1,101)[:,None]

    if kernel == "gaussian":
        Kernel = GaussianKernel
    else:
        Kernel = GammaExponentialKernel
        exact = False

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
    var = model.post_var(Y, Z, V, doZ, reg = reg, latent = True, diag = True).detach()


    """ Compute out of sample metrics """
    z_quantiles = Normal(0, 1).icdf(1-(1-quantiles)/2)
    posterior_fraction = ((EYdoZ - mean).abs() <= var**0.5 @ z_quantiles.T).float()
    rmse = ((EYdoZ - mean)**2).mean()**0.5

    return {"name" : "bayesimp_optmu={0}_exact={1}".format(optimise_mu,exact),
            "rmse" : rmse, 
           "cal_levels" : quantiles,
           "post_levels" : posterior_fraction,
           }