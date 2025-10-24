def marginal_posterior_mean(Y_train, X_train, Z_train, z_test, 
                            kernel_params_x, kernel_params_z, 
                            y_noise, z_noise, reg=1e-8):
    """
    vectorised for multiple (W*,Z*), fixed A* = [a*,...,a*]^T
    returns: E[Y|a,w,z,D]
    """
    n = len(Y_train)
    K_xx = get_gram_gaussian(X_train, X_train, kernel_params_x)
    K_zz,k_z = get_gram_gaussian(Z_train, Z_train, kernel_params_z), get_gram_gaussian(z_test, Z_train, kernel_params_z)
    I = torch.eye(n)
    
    alpha_y = torch.linalg.solve(K_xx + (y_noise+reg)*I,Y_train)
    alpha_phi = torch.linalg.solve(K_zz + (z_noise+reg)*I, K_xx @ alpha_y)
    
    return k_z @ alpha_phi

def marginal_posterior_covariance(Y_train,X_train,Z_train,
                                       z1,z2, noise_y, noise_phi,
                                       kernel_params_x,kernel_params_z,K_xx_tilde, eigsum, latent = False):
    n = len(X_train)
    
    # get posterior variance of phi(x)
    kernel_params_phi = kernel_params_z
    if z2!=[]:
        phi_post_cov = GP_post_var(Z_train, torch.cat((z1,z2),0), kernel_params_phi,noise_phi, latent)[1,0]
    else:
        phi_post_cov = GP_post_var(Z_train, z1, kernel_params_phi, noise_phi, latent)
    
     # get gram matrices
    K_xx,K_zz = (get_gram_gaussian(X_train, X_train, kernel_params_x),
                 get_gram_gaussian(Z_train, Z_train, kernel_params_z))
    k_z1 = get_gram_gaussian(z1, Z_train, kernel_params_z)
    if z2!=[]:
        k_z2 = get_gram_gaussian(z2, Z_train, kernel_params_z)
    else:
        k_z2= k_z1
    
    # compute matrix-vector products
    K_phi = K_zz+noise_phi*torch.eye(n)
    K_y = K_xx+noise_y*torch.eye(n)
    alpha_y = torch.linalg.solve(K_y, Y_train)
    B = torch.linalg.solve(K_y, K_xx_tilde)
    alpha1 = torch.linalg.solve(K_phi,K_xx)
    alpha2 = torch.linalg.solve(K_phi,k_z2.T)
    DDKxx =  torch.linalg.solve(K_y, K_xx)
    V1 = phi_post_cov*(alpha_y.T @ K_xx_tilde @ alpha_y).view(1,)
    V2 = phi_post_cov*(eigsum - torch.trace(B))
    V3 = k_z1 @ alpha1 @ (torch.eye(n) - DDKxx) @ alpha2+noise_y*(not latent)*(z2==[])*torch.eye(len(z1))
    return V1+V2+V3

def BayesImp_mean(Y,X,Z,Ztest,kernel_params_r,kernel_params_z,noise_r,noise_z, mc_approx = False, mc_samples = 10**3, nu = 1):
    n = len(Y)
    
    # getting kernel matrices
    K_xx,K_zz,k_ztest = (get_gram_gaussian(X, X, kernel_params_r),
                         get_gram_gaussian(Z, Z, kernel_params_z),
                         get_gram_gaussian(Ztest, Z, kernel_params_z))
    if mc_approx:
        R_xx = get_smoothed_Gaussian_kernel(X, kernel_params_r, mc_approx = True, mc_samples = mc_samples, nu = nu)
    else:
        R_xx = get_nuclear_kernel(X,X, kernel_params_r, nu = nu)
    K_z = K_zz+noise_z*torch.eye(n)
    R_x = R_xx+noise_r*torch.eye(n)
    
    A_z = torch.linalg.solve(K_z,k_ztest.T).T
    alpha_r = torch.linalg.solve(R_x,Y)
    
    return  A_z @ R_xx @ alpha_r

def BayesImp_var(Y,X,Z,Ztest,kernel_params_r,kernel_params_z,noise_r,noise_z, mc_approx = False, mc_samples = 10**3, nu = 1):
    
    n = len(Y)
    
    # getting kernel matrices
    K_xx,K_zz,k_ztest = (get_gram_gaussian(X, X, kernel_params_r),
                         get_gram_gaussian(Z, Z, kernel_params_z),
                         get_gram_gaussian(Ztest, Z, kernel_params_z))
    if mc_approx:
        R_xx = get_smoothed_Gaussian_kernel(X, kernel_params_r, mc_approx = True, mc_samples = mc_samples, nu = nu)
    else:
        R_xx = get_nuclear_kernel(X,X, kernel_params_r, nu = nu)
    K_z = K_zz+noise_z*torch.eye(n)
    K_x = K_xx+noise_r*torch.eye(n)
    R_x = R_xx+noise_r*torch.eye(n)
    R_xx_bar = R_xx - R_xx @ torch.linalg.solve(R_x,R_xx)
    k_post_ztest_approx = (k_ztest @ k_ztest.T - k_ztest @ torch.linalg.solve(K_z,k_ztest.T))
    
    # computing matrix vector products
    alpha_z = torch.linalg.solve(K_z,k_ztest.T)
    alpha_y = torch.linalg.solve(K_x,Y)
    KinvR = torch.linalg.solve(K_xx+torch.eye(n)*1e-3,R_xx)
    
    V1 = alpha_z.T @ R_xx_bar @ alpha_z
    V2 = alpha_y.T @ R_xx @ KinvR @ KinvR @ alpha_y * k_post_ztest_approx
    V3 = torch.trace(torch.linalg.solve(K_xx+torch.eye(n)*1e-3, R_xx_bar @ KinvR)) * k_post_ztest_approx
    
    return V1+V2+V3

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