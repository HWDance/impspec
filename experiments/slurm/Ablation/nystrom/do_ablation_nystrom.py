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
         marginal_loss = False, retrain_hypers = False, features = 100, samples = 1000):
    
    """ Fixed configs """
    default_nu = 1.0
    cal_nulist = 2**torch.linspace(-4,4,5)
    quantiles = torch.linspace(0,1,101)
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
    
    """ Get Posterior samples """
    mean = model.post_mean(Y, Z, V, doZ).detach()
    EYdoZ_samples, EVdoZ_samples = model.nystrom_sample(Y,V,Z,doZ, 
                                                            reg = reg, 
                                                            features = features, 
                                                            samples = samples)

    """ Compute out of sample metrics """
    upper_quantiles = 1-(1-quantiles)/2
    lower_quantiles = (1-quantiles)/2
    u = (upper_quantiles*(samples-1)).int()
    l = (lower_quantiles*(samples-1)).int()
    EY_u = EYdoZ_samples.T.sort(1)[0][:,u]
    EY_l = EYdoZ_samples.T.sort(1)[0][:,l]

    posterior_fraction = ((EY_u>=EYdoZ)*(EY_l<=EYdoZ)).float()   
    rmse = ((EYdoZ - mean)**2).mean()**0.5

    return {"name" : "nystrom_cal={0}_split={1}".format(calibrate, sample_split),
            "rmse" : rmse, 
           "cal_levels" : quantiles,
           "post_levels" : posterior_fraction,
           }