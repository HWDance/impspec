import torch

def get_CV_splits(X,folds=[]):
    """
    X: nxd matrix
    folds: # CV folds 
    
    Returns: list of folds x training and validation sets
    """
    
    n = len(X)
    if folds==[]:
        folds = n
    n_per_fold = int(n/folds+1-1e-10) # rounds up except if exact integer
    row_count = torch.linspace(0,n-1,n) 
    train_val_sets = list()
    
    for i in range(folds):
        test_inds = ((row_count>= n_per_fold*i)*(row_count<n_per_fold*(i+1)))>0
        train_inds = test_inds==0
        train_val_sets.append([X[test_inds],X[train_inds]])
    
    return train_val_sets

def get_CV_loss(Y,X,Z, 
                kernel_params_x, kernel_params_z, 
                y_noise, z_noise,func,folds = []):
    
    # Randomising order
    """
    n=len(Y)
    inds = torch.randperm(n)
    Y,X,Z = (Y[inds],
                X[inds],
                Z[inds])
    """
    # getting splits
    Y_splits = get_CV_splits(Y,folds)
    X_splits = get_CV_splits(X,folds)
    Z_splits = get_CV_splits(Z,folds)
    
    loss = 0
    
    for i in range(folds):
        mu_i = func(Y_splits[i][1], X_splits[i][1], Z_splits[i][1], Z_splits[i][0],
                                kernel_params_x, kernel_params_z, 
                                y_noise, z_noise)
        loss += ((Y_splits[i][0]-mu_i)**2).sum()
        
    return loss.sqrt()/len(Y)