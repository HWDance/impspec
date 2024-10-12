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
def main(seed, n, n_int, int_samples = 10**5, noise = 1.0, int_scale = 4, front_door = False, minimise = False): 
    
    torch.manual_seed(seed)

    """ CBO configs """
    n_iter = 10
    xi = 0.0
    update_hyperparameters = True
    update_interval = 5
    noise_init = -10.0
    cbo_reg = 1e-3
        
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
               int_max = int_scale*noise
                )

    """ Variable definitions """
    A = B
    V = C
    if front_door:
        W = B
        doA = vals
        doW = torch.zeros((1,1))     
    else:
        W = D
        doA = torch.zeros((1,1))
        doW = vals
        
        
    """ Get CBO prior kernel """
    def mean(X):
        return torch.zeros((len(X),1))
    
    def var(X, diag = False):
        return torch.zeros((len(X),1))

            
    rbf_kernel = GaussianKernel(lengthscale=torch.tensor([0.1]).requires_grad_(True), 
                            scale=torch.tensor([1.0]).requires_grad_(True)) # 5.0 originally used
    cbo_kernel = CausalKernel(
        estimate_var_func=var,
        base_kernel=rbf_kernel,
        add_base_kernel=True
    )

    """ Run CBO """
    # Define a grid of intervention points and precompute E[Y|do(x)]
    doX = vals
    
    # Random search for first intervention point
    torch.manual_seed(seed)
    start = torch.randint(0,len(vals)-1,(1,))[0]
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
                                                        update_interval = update_interval,
                                                        xi = xi, 
                                                        print_ = False, 
                                                        minimise = minimise,
                                                        noise_init = noise_init,
                                                        reg = cbo_reg)
    
    """ Returning outputs """
    if front_door:
        obs_data = [A,Y]
        int_data = [doA,EYdoX]
    else:
        obs_data = [W,Y]
        int_data = [doW,EYdoX]   
    return {"name" : "naive_cbo_frontdoor={0}_minimise={1}".format(front_door, minimise),
            "doXeval" : doXeval,
            "EYdoXeval" : EYdoXeval
               }