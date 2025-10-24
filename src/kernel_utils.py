import torch

"""
Eigendecomposition helpers
"""

def kernel_approximations(X_train, kernel_params, mc_approx = True, mc_samples = 10**3, dist = "normal"):
    
    if mc_approx:
        if dist == "unif":
            measure = torch.distributions.uniform.Uniform(X_train.min(),X_train.max())
        else:
            measure = torch.distributions.normal.Normal(0,1)            
        
        X_sample = measure.sample((mc_samples,len(X_train.T)))
        K_s = get_gram_gaussian(X_train, X_sample, kernel_params)
        K_ss = get_gram_gaussian(X_sample, X_sample, kernel_params)
        K_xx_tilde = K_s @ K_s.T/mc_samples
        # Get eigenvalues
        eigs = torch.linalg.eig(K_ss)[0].real/mc_samples
        
    return K_xx_tilde, K_ss, eigs


def get_Gaussian_kernel_eigs(X_train,kernel_params, mc_approx = True, mc_samples = 10**3, nu = 1): # 1d case only for P(x) = N(0,1)
    
    if mc_approx: 
        measure = torch.distributions.normal.Normal(0,nu)            
        X_sample = measure.sample((mc_samples,len(X_train.T)))
        K_sample = get_gram_gaussian(X_sample, X_sample, kernel_params)
        eigs = torch.linalg.eig(K_sample)[0].real/mc_samples
    
    else: 
        a = 1/4
        b = 1/(2*kernel_params[1]**2)
        c = (a**2+2*a*b).sqrt()
        A = a + b + c
        B = b/A
        eigs = kernel_params[0]**2*(2*a/A).sqrt()*B**(torch.linspace(0,mc_samples-1,mc_samples))
    return eigs


"""
Returns median heuristic lengthscale for Gaussian kernel
"""
def median_heuristic(X, per_dimension = False):
    if not per_dimension:
        # Median heurstic for inputs
        Dist = torch.cdist(X,X, p = 2.0)**2
        Lower_tri = torch.tril(Dist, diagonal=-1).view(len(X)**2).sort(descending = True)[0]
        Lower_tri = Lower_tri[Lower_tri!=0]
        return (Lower_tri.median()/2).sqrt()
        
    else:
        medheur = torch.zeros(X.shape[1])
        for i in range(X.shape[1]):
            Dist = torch.cdist(X[:,i:i+1],X[:,i:i+1], p = 2.0)**2
            Lower_tri = torch.tril(Dist, diagonal=-1).view(len(X)**2).sort(descending = True)[0]
            Lower_tri = Lower_tri[Lower_tri!=0]
            medheur[i] = (Lower_tri.median()/2).sqrt()
        return medheur
        


