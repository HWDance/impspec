# imports
import torch
import sys
from pathlib import Path

path = Path.cwd().parents[3]
if str(path) not in sys.path:
    sys.path.append(str(path))

from src.kernels import *
from src.dgps import *
from src.CBO import *
from src.kernel_utils import *

# main
def main(seed, n, n_int): 
    
    torch.manual_seed(seed)

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
        
    """ Get CBO prior kernel """
    def mean(X):
        return torch.zeros((len(X),1))
    
    def var(X, diag = False):
        return torch.zeros((len(X),1))

            
    #medheur = median_heuristic(statin[:,None].reshape(n_int,int_samples).mean(1)[:,None])
    rbf_kernel = GaussianKernel(lengthscale=torch.tensor([0.1]).requires_grad_(True), 
                            scale=torch.tensor([5.0]).requires_grad_(True))
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
                                                        mean = lambda x : torch.zeros(len(x),1),
                                                        X_test = doX, 
                                                        Y_test = EYdoX, 
                                                        n_iter = n_iter, 
                                                        update_hyperparameters = update_hyperparameters,
                                                        xi = xi, 
                                                        print_ = False, 
                                                        minimise = True,
                                                        noise_init = noise_init,
                                                        reg = cbo_reg)

    return {"name" : "naive bo",
            "doXeval" : doXeval, 
           "EYdoXeval" : EYdoXeval,
           }