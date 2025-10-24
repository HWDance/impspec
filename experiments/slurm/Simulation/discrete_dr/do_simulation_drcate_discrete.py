# imports
import torch
from torch.distributions import Normal
import matplotlib.pyplot as plt
from functools import partial
import sys
from pathlib import Path

# imports for discrete DR
import numpy as np
import pandas as pd
import doubleml as dml
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.base import clone
import patsy 

# change import path
path = Path.cwd().parents[3]
if str(path) not in sys.path:
    sys.path.append(str(path))
    
from src.dgps import *

quantiles = torch.linspace(0,1,101)

def main(seed, n=100, n_int=100, int_samples=10**5, noise=1.0, int_scale = 1):

    """ Setting method """
    method = "CATE_backdoor_doD_bfixed"

    """ Drawing data """
    A,B,C,D,E,Y,vals,EYdoX = Simulation(n,n_int, 
               mc_samples_EYdoX = int_samples, 
               seed = seed, 
               draw_EYdoX = True, 
               noise = noise,
               method = method, 
               int_min=-int_scale*noise, 
               int_max = int_scale*noise,
               discrete_D = True,
               fix_b = False,
                )

    """ DML implementation """
    # Set up basic model: Specify variables for data-backend
    features_base = ['C']
    
    # Initialize DoubleMLData (data-backend of DoubleML)
    Data = pd.DataFrame((D+1)/2, columns=['D'])
    Data['C'] = C
    Data['B'] = B
    Data['Y'] = Y
    data_dml_base = dml.DoubleMLData(Data,
                                     y_col='Y',
                                     d_cols='D',
                                     x_cols=features_base)
    
    # Random Forest (IRM)
    randomForest = RandomForestRegressor(n_estimators=100,max_features=3, max_depth=3, min_samples_leaf=3)
    randomForest_class = RandomForestClassifier(n_estimators=100,max_features=3, max_depth=3, min_samples_leaf=3)
    
    np.random.seed(1)
    dml_irm_forest = dml.DoubleMLIRM(data_dml_base,
                                     ml_g = randomForest,
                                     ml_m = randomForest_class,
                                     trimming_threshold = 0.01,
                                     n_folds = 5,
                                     score = "ATE",
                                     n_rep = 1)
    
    dml_irm_forest.fit(store_predictions=True)

    # Getting CATE using B-splines (order 3)
    design_matrix = patsy.dmatrix("bs(x, df=5, degree=2)", {"x": Data["B"]})
    spline_basis = pd.DataFrame(design_matrix)
    cate = dml_irm_forest.cate(spline_basis)

    new_data = {"x": np.linspace(0, 0, 1)}
    spline_grid = pd.DataFrame(patsy.build_design_matrices([design_matrix.design_info], new_data)[0])

    posterior_fraction = torch.zeros(len(quantiles))
    posterior_fraction[0],posterior_fraction[-1] = 0,1
    for i in range(1,len(quantiles)-1):
            df_cate = cate.confint(spline_grid, level=quantiles[i].item(), joint=False, n_rep_boot=1000)
            posterior_fraction[i] = ((0 >= torch.tensor(df_cate[df_cate.columns[0]].iloc[0])) *
                                 (0 <= torch.tensor(df_cate[df_cate.columns[2]].iloc[0])) )*1
    mu = torch.tensor(df_cate['effect'].iloc[0])
    
    """ Compute out of sample metrics """
    rmse = mu.abs()

    """ Returning outputs """
    obs_data = [D,Y]
    int_data = [vals,EYdoX] 
        
    return {"name" : "dr",
                "rmse" : rmse, 
                "cal_levels" : quantiles,
                "post_levels" : posterior_fraction,
                "post_moments" : [mu,[]],
                "obs_data" : obs_data,
                "int_data" : int_data,
                "doXeval" : [],
                "EYdoXeval" : []
               }