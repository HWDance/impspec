import torch
from torch.distributions import Normal, StudentT, Gamma, Uniform
from math import pi    

def f_x(Z,coefs):
    return torch.sin(Z*coefs)

def f_y(X,coefs):
    return (torch.sin(X)*coefs).T.sum(0)

def Abelation(n, ntest, d, noise_variance, doZlower = 0, doZupper = 1, mc_samples_EYdoZ = 10**4, seed = 0):

    torch.manual_seed(seed)
    
    coefs_v = 10*torch.linspace(1,4,d).view(1,d)
    coefs_y = 1/torch.linspace(1,d,d).view(d,)
    
    Z = Uniform(0,1).sample((n,1))
    doZ = torch.linspace(doZlower,doZupper,ntest).view(ntest,1)
    fv = f_x(Z,coefs_v)
    noise_distribution = Normal(0,(noise_variance*fv.var(0))**0.5)
    V =  fv+noise_distribution.sample((n,))
    fy = f_y(V,coefs_y)
    Y = Normal(fy,(noise_variance*fy.var())**0.5).sample()
    
    # Grid-points to approximate true E[Y|do(Z)]
    VdoZ = (f_x(doZ,coefs_v)).T[:,:,None] @ torch.ones(mc_samples_EYdoZ).view(1,mc_samples_EYdoZ) + noise_distribution.sample((mc_samples_EYdoZ,ntest)).T
    EYdoZ = (f_y(VdoZ.T,coefs_y)).mean(1).view(ntest,1)
    YdoZ = Normal(f_y(VdoZ[...,0].T,coefs_y),(noise_variance*fy.var())**0.5).sample().view(ntest,1)

    return Z, V, Y, doZ, YdoZ, EYdoZ

def BayesIMP_Abelation(n,ntest, sigma_x,sigma_y,sigma_t, get_ET = True, mc_samples = 10**3):
    X = Normal(0,sigma_x).sample((n+ntest,1))
    Y = X*torch.cos(pi*X) + Normal(0,sigma_y).sample((n+ntest,1))
    T = 0.5*Y*torch.cos(Y) + Normal(0,sigma_t).sample((n+ntest,1))

    Xtrain, Ytrain, Ttrain = X[:n], Y[:n], T[:n]
    Xtest, Ytest, Ttest =  X[n:], Y[n:], T[n:]

    if get_ET:
        Ysample = (Xtest*torch.cos(pi*Xtest)) @ torch.ones((1,mc_samples)) + Normal(0,sigma_y).sample((ntest,mc_samples))
        ET = (0.5*Ysample*torch.cos(Ysample)).mean(1) + Normal(0,sigma_t).sample((ntest,mc_samples)).mean(1)
        
    return Xtrain, Ytrain, Ttrain, Xtest, Ytest, Ttest, ET


def STATIN_PSA(samples, seed = 0, gamma = False):
    torch.manual_seed(seed)

    age = Uniform(15,75).sample((samples,))
    bmi = Normal(27-0.01*age, 0.7**0.5).sample()
    aspirin = torch.sigmoid(-8 + 0.1*age + 0.03*bmi)
    statin = torch.sigmoid(-13 + 0.1*age + 0.2*bmi)
    cancer = torch.sigmoid(2.2 - 0.05*age + 0.01*bmi - 0.04*statin + 0.02*aspirin)
    if gamma:
        psa = Gamma(100, 20/(6.8 + 0.04*age - 0.15*bmi - 0.60*statin + 0.55*aspirin + cancer)).sample()
    else:
        psa = 5*Normal(6.8 + 0.04*age - 0.15*bmi - 0.60*statin + 0.55*aspirin + cancer, 0.4**0.5).sample()
    return age, bmi, aspirin, statin, cancer, psa


def PSA_VOL(samples, seed = 0, psa = []):
    
    torch.manual_seed(seed)
    
    # Estimated in Kato et al (2008)
    def get_vol(psa):
        return 3.476 + 0.302*psa
    r2 = 0.332**2

    # Inferred from Kato et al (2008) results
    PSA_dist = Gamma(2,0.2)
    error_dist = StudentT(3.5,0,1)

    # Sampling
    if psa == []:
        psa = PSA_dist.sample((samples,))
    fvol = get_vol(psa)
    vol = (fvol + error_dist.sample((samples,))*(fvol.var()**0.5*(1-r2)/r2)**0.5).abs()

    return psa, fvol, vol