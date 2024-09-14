# imports
import torch
from torch.distributions import Normal
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from functools import partial
from copy import deepcopy

path = os.path.dirname(os.getcwd())
os.chdir(path)

from src.causalKLGP import *
from src.kernels import *
from src.GP_utils import *
from src.kernel_utils import *
from src.dgps import *

def run_experiment(n,d,p,noise, niter = 500, learn_rate = 0.1, train_lsA = True, cal_latent = True, traincalmodel = False, kernel = "gaussian"):

    # Draw data 
    Z, V, Y, doZ, YdoZ, EYdoZ = get_data()
    
    # Specify storage objects
    rmse = []
    calibration = []
    
    # Initialise model

    # Training + calibration

    # Posterior moments

    # Out of sample metrics