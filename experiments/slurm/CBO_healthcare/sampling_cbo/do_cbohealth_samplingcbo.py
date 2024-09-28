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
from src.CBO import *
from src.kernel_utils import *

# main
def main(seed, n, n_int, two_datasets = True): 
    
    torch.manual_seed(seed)

    """ baselinegp configs"""
    niter = 1000
    learn_rate = 0.1
    single_kernel = False
    force_PD = True
    reg = 1e-3
    error_samples = 100
    gp_samples = 100
    Kernel = GaussianKernel

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
    model = baselineGP(Kernel_A = Kernel, 
                   Kernel_V = Kernel, 
                   Kernel_Z = [],
                   dim_A = A.size()[1], 
                   dim_V = V[1].size()[1], 
                   single_kernel = single_kernel,
                   scale_V_init = Y.var()**0.5/2,
                   noise_Y_init = torch.log(Y.var()/4))
    
    """ Train model """
    model.train(Y,A,V,niter,learn_rate, force_PD = force_PD)

    """ Get CBO prior kernel """
    def mean(X):
        EYdoX_samples, EVdoX_samples = model.marginal_post_sample(Y,V,A,X, 
                                                            reg = reg, 
                                                            error_samples = error_samples, 
                                                            gp_samples = gp_samples)
        return EYdoX_samples.mean(1)[:,None]
            
    def var(X, diag = False):
        EYdoX_samples, EVdoX_samples = model.marginal_post_sample(Y,V,A,X, 
                                                            reg = reg, 
                                                            error_samples = error_samples, 
                                                            gp_samples = gp_samples)
        return EYdoX_samples.var(1)[:,None]

            
    medheur = median_heuristic(statin[:,None].reshape(n_int,int_samples).mean(1)[:,None])
    rbf_kernel = GaussianKernel(lengthscale=torch.tensor([medheur]).requires_grad_(True), 
                            scale=torch.tensor(Y.var()**0.5/2).requires_grad_(True))
    cbo_kernel = CausalKernel(
        estimate_var_func=var,
        base_kernel=rbf_kernel,
        add_base_kernel=True
    )

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

    return {"name" : "sampling_cbo",
            "doXeval" : doXeval, 
           "EYdoXeval" : EYdoXeval,
           }