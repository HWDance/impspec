# imports
import torch
import sys
from pathlib import Path

path = Path.cwd().parents[3]
if str(path) not in sys.path:
    sys.path.append(str(path))

from src.baselineGP import *
from src.kernels import *
from src.dgps import *

# main
def main(seed, n,ntest,d,noise, niter = 500, learn_rate = 0.1, error_samples = 10**2, gp_samples = 10**2,kernel = "gaussian"):
    
    torch.manual_seed(seed)

    """ Fixed configs """
    Kernel = GaussianKernel
    single_kernel = False
    quantiles = torch.linspace(0,1,101)
    reg = 1e-2
    force_PD = True

    if kernel == "gaussian":
        Kernel = GaussianKernel
    else:
        Kernel = GammaExponentialKernel
    
    """ Draw data """
    Z, V, Y, doZ, YdoZ, EYdoZ = Abelation(n, ntest, d, noise, doZlower = 0, doZupper = 1, mc_samples_EYdoZ = 10**4, seed = seed) 
    
    """ Initialise model """
    model = baselineGP(Kernel_A = Kernel, 
                   Kernel_V = Kernel, 
                   dim_A = Z.size()[1], 
                   dim_V = V.size()[1], 
                   single_kernel = single_kernel)

    """ Train model """
    model.train(Y,Z,V,niter,learn_rate, force_PD = force_PD, reg = reg)
    
    """ Get Posterior mean """
    EYdoZ_samples, EVdoZ_samples = model.marginal_post_sample(Y,V,Z,doZ, 
                                                            reg = reg, 
                                                            error_samples = error_samples, 
                                                            gp_samples = gp_samples)

    """ Compute out of sample metrics """
    upper_quantiles = 1-(1-quantiles)/2
    lower_quantiles = (1-quantiles)/2
    u = (upper_quantiles*(gp_samples-1)).int()
    l = (lower_quantiles*(gp_samples-1)).int()
    EY_u = EYdoZ_samples.sort(1)[0][:,u]
    EY_l = EYdoZ_samples.sort(1)[0][:,l]

    posterior_fraction = ((EY_u>=EYdoZ)*(EY_l<=EYdoZ)).float()   
    rmse = ((EYdoZ - EYdoZ_samples.mean(1).reshape(ntest,1))**2).mean()**0.5

    return {"name" : "baselinegp",
            "rmse" : rmse, 
           "cal_levels" : quantiles,
           "post_levels" : posterior_fraction,
           "post_samples" : [EYdoZ_samples],
            "obs_data" : [Z,Y],
            "int_data" : [doZ,EYdoZ]
           }