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
from src.CBO import *

# main
def main(seed, n, n_int, two_datasets = True, niter = 500, learn_rate = 0.1,
         calibrate = True, sample_split = True, marginal_loss = False, retrain_hypers = False,
        add_base_kernel_bo = False): 
    
    torch.manual_seed(seed)
    
    """ causalklgp configs """
    default_nu = 1.0
    cal_nulist = 2**torch.linspace(-4,4,5)
    quantiles = torch.linspace(0,1,101)[:,None]
    reg = 1e-3
    Kernel = GaussianKernel
    force_PD = True

    """ CBO configs """
    int_samples = 10**5
    n_iter = 20
    xi = 0.0
    update_hyperparameters = False
    noise_init = -10.0
    cbo_reg = 1e-3
    
    """ Draw int data """
    dostatin = torch.linspace(0,1,n_int)
    age, bmi, aspirin, statin, cancer, psa = STATIN_PSA(int_samples, 
                                                        seed = seed, 
                                                        gamma = False, 
                                                        interventional_data = True, 
                                                        dostatin = dostatin)
    psa,fvol, vol = PSA_VOL(psa = psa)  

    """ Draw training data"""
    age_, bmi_, aspirin_, statin_, cancer_, psa_ = STATIN_PSA(n, 
                                                              seed = seed, 
                                                              gamma = False, 
                                                              interventional_data = False, 
                                                              dostatin=[])
    if two_datasets:
        age_2, bmi_2, aspirin_2, statin_2, cancer_2, psa_2 = STATIN_PSA(n, 
                                                          seed = seed+1, 
                                                          gamma = False, 
                                                          interventional_data = False, 
                                                          dostatin=[])
        psa_2, fvol_2, vol_2 = PSA_VOL(psa = psa_2)
        A = torch.column_stack((age_, bmi_, aspirin_, statin_))
        V = [psa_.reshape(len(psa_),1),psa_2.reshape(len(psa_),1)]
        Y = vol_2
        
    else:
        psa_, fvol_, vol_ = PSA_VOL(psa = psa_)
        A = torch.column_stack((age_, bmi_, aspirin_, statin_))
        V = [psa_.reshape(len(psa_),1), psa_.reshape(len(psa_),1)]
        Y = vol_
    
    """ Initialise model """
    model = causalKLGP(Kernel_A = Kernel, 
                   Kernel_V = Kernel, 
                   dim_A = A.size()[1], 
                   dim_V = V[1].size()[1], 
                   samples = 10**5,
                   scale_V_init = Y.var()**0.5/2,
                   noise_Y_init = torch.log(Y.var()/4)
                  )
    
    """ Train + Calibrate model """
    if calibrate:
        Post_levels, Calibration_losses = model.frequentist_calibrate(Y, V, A, dostatin[:,None],
                                                                     niter = niter, 
                                                                     learn_rate = learn_rate,
                                                                     bootstrap_replications = bootreps,
                                                                     nulist = cal_nulist,                                                                                           sample_split = sample_split,
                                                                     marginal_loss = marginal_loss,
                                                                     retrain_hypers = retrain_hypers,
                                                                     average_doA = True,
                                                                     intervention_indices = [3],
                                                                     force_PD = force_PD,
                                                                     reg = reg
                                                                    )
        best_ind = torch.where(Calibration_losses == Calibration_losses.min())[0][0]
        nu_best = cal_nulist[best_ind]
    else:
        nu_best = default_nu
        
    if (not calibrate) or sample_split: 
        model.train(Y, A, V, niter, learn_rate, force_PD = force_PD, reg = reg)   
        
    """ Get posterior funcs and CBO prior kernel """
    def mean(X):
        doA = X.reshape(len(X),1)
        return model.post_mean(Y,A,V,doA,
                               reg = reg, 
                               average_doA = True, 
                               intervention_indices = [3]) 
    
    def cov(X, Z, diag = False):
        doA = X.reshape(len(X),1)
        doA2 = Z.reshape(len(Z),1)
        return model.post_var(Y,A,V,doA,doA2,
                              reg = reg,
                              nu = nu_best, 
                              average_doA = True, 
                              intervention_indices = [3], 
                              diag = diag)

    cbo_kernel = CBOPriorKernel(cov)

    """ Run CBO """
    # Define a grid of intervention points and precompute E[Y|do(x)]
    doX = dostatin[:,None]
    EYdoX = fvol.reshape(n_int,int_samples).mean(1)[:,None]

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
                                                        minimise = True,
                                                        noise_init = noise_init,
                                                        reg = cbo_reg)

    return {"name" : "causalklgp_cal={0}_split={1}".format(calibrate, sample_split),
            "doXeval" : doXeval, 
           "EYdoXeval" : EYdoXeval,
           }