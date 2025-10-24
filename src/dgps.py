import torch
from torch.distributions import Normal, StudentT, Gamma, Uniform
from math import pi    

def f_x(Z,coefs):
    return torch.sin(Z*coefs)

def f_y(X,coefs):
    return (torch.sin(X)*coefs).T.sum(0)

def Abelation(n, 
              ntest, 
              d, 
              noise_variance, 
              doZlower = 0, 
              doZupper = 1, 
              mc_samples_EYdoZ = 10**4, 
              seed = 0, 
              draw_EYdoZ = True):

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
    
    if draw_EYdoZ:
        # Grid-points used to approximate true E[Y|do(Z)]
        VdoZ = (f_x(doZ,coefs_v)).T[:,:,None] @ torch.ones(mc_samples_EYdoZ).view(1,mc_samples_EYdoZ) + noise_distribution.sample((mc_samples_EYdoZ,ntest)).T
        EYdoZ = (f_y(VdoZ.T,coefs_y)).mean(1).view(ntest,1)
        YdoZ = Normal(f_y(VdoZ[...,0].T,coefs_y),(noise_variance*fy.var())**0.5).sample().view(ntest,1)
    
        return Z, V, Y, doZ, YdoZ, EYdoZ
    else: 
        return Z, V, Y

def Simulation(n,ntest, 
               mc_samples_EYdoX = 10**4, 
               seed = 0, 
               draw_EYdoX = True, 
               noise = 1.0, 
               method = "backdoor_", 
               int_min=None, 
               int_max = None, 
               discrete_D = False,
               fix_b = False):
    """
    method: CATE_backdoor_doD_b, ATT_frontdoor_doB_b, CATE_backdoor_doD_c 
    """

    torch.manual_seed(seed)
    
    m = mc_samples_EYdoX
    dist = Normal(0,noise)
    
    U1 = dist.sample((n,1))
    U2 = dist.sample((n,1))
    F =  dist.sample((n,1))
    A = F**2 + U1 + dist.sample((n,1))
    B = U2 + dist.sample((n,1))
    if fix_b:
        B = B*0
    C = torch.exp(-B) + dist.sample((n,1))
    D = torch.exp(-C)/10 + dist.sample((n,1))
    if discrete_D:
        D = 2*(D>=0)-1
    E = torch.cos(A) + C/10 + dist.sample((n,1))
    Y = torch.cos(D) + torch.sin(E) + U1 + U2*dist.sample((n,1))

    if draw_EYdoX:

        if method == "ATT_frontdoor_doB_b":

            # Get intervention values
            b_vals = torch.linspace(int_min,int_max,ntest)[:,None]

            # Sample from p(c|b)
            C_ = torch.exp(-b_vals) + dist.sample((1,m)) # ntest x m

            # Sample from p(A)
            U1_ = dist.sample((1,m)) # 1 x n
            F_ =  dist.sample((1,m)) # 1 x n
            A_ = F_**2 + U1_ + dist.sample((1,m)) # 1 x n

            # Compute expectations
            EY1 = torch.cos(torch.exp(-C_)/10 + dist.sample((1,m))).mean(1)
            EY2 = torch.sin(torch.cos(A_) + C_/10 + dist.sample((1,m))).mean(1)
            EY = EY1 + EY2

            return A,B,C,D,E,Y,b_vals,EY.reshape(len(EY),1)          

        if method == "CATE_backdoor_doD_b":

            # get conditioning values
            b_vals = torch.linspace(int_min,int_max,ntest)[:,None]

            # Sample from p(c|b)
            C_ = torch.exp(-b_vals) + dist.sample((1,m)) # ntest x m

            # Sample from p(A)
            U1_ = dist.sample((1,m)) # 1 x n
            F_ =  dist.sample((1,m)) # 1 x n
            A_ = F_**2 + U1_ + dist.sample((1,m)) # 1 x n

            # Compute expectations
            EY1 = torch.cos(torch.zeros(1)) # do(D=0)
            EY2 = torch.sin(torch.cos(A_) + C_/10 + dist.sample((1,m))).mean(1)
            EY = EY1 + EY2

            return A,B,C,D,E,Y,b_vals,EY.reshape(len(EY),1)
        
        if method == "CATE_backdoor_doD_bfixed":

            # get conditioning values
            d_vals = torch.linspace(int_min,int_max,ntest)[:,None]
            
            # Sample from p(c|b=0)
            C_ = torch.exp(-torch.zeros((1,1))) + dist.sample((1,m)) # 1 x m

            # Sample from p(A)
            U1_ = dist.sample((1,m)) # 1 x n
            F_ =  dist.sample((1,m)) # 1 x n
            A_ = F_**2 + U1_ + dist.sample((1,m)) # 1 x n

            # Compute expectations
            EY1 = torch.cos(d_vals) # do(D=0)
            EY2 = torch.sin(torch.cos(A_) + C_/10 + dist.sample((1,m)))
            EY = (EY1 + EY2).mean(1)

            return A,B,C,D,E,Y,d_vals,EY.reshape(len(EY),1)

        if method == "CATE_backdoor_doD_c":

            # get conditioning values
            c_vals = torch.linspace(int_min,int_max,ntest)[:,None]

            # Sample from p(A)
            U1_ = dist.sample((1,m)) # 1 x n
            F_ =  dist.sample((1,m)) # 1 x n
            A_ = F_**2 + U1_ + dist.sample((1,m)) # 1 x n

            # Compute expectations
            EY1 = torch.cos(torch.zeros(1)) # do(D=0)
            EY2 = torch.sin(torch.cos(A_) + c_vals/10 + dist.sample((1,m))).mean(1)
            EY = EY1 + EY2

            return A,B,C,D,E,Y,c_vals,EY.reshape(len(EY),1)
    
    else:
        return A, B, C, D, E, Y
    
    


def STATIN_PSA(samples = 10**4, seed = 0, gamma = False, 
               interventional_data = False, dostatin=[]):
    
    torch.manual_seed(seed)

    if interventional_data:
        statin = dostatin.repeat_interleave(samples)
        samples *= len(dostatin)

    age = Uniform(15,75).sample((samples,))
    bmi = Normal(27-0.01*age, 0.7**0.5).sample()
    aspirin = torch.sigmoid(-8 + 0.1*age + 0.03*bmi)
    if not interventional_data:
        statin = torch.sigmoid(-13 + 0.1*age + 0.2*bmi)
    cancer = torch.sigmoid(2.2 - 0.05*age + 0.01*bmi - 0.04*statin + 0.02*aspirin)
    if gamma:
        psa = Gamma(100, 20/(6.8 + 0.04*age - 0.15*bmi - 0.60*statin + 0.55*aspirin + cancer)).sample()
    else:
        psa = Normal(6.8 + 0.04*age - 0.15*bmi - 0.60*statin + 0.55*aspirin + cancer, 0.4**0.5).sample()
    
    return age, bmi, aspirin, statin, cancer, psa

def PSA_VOL(samples = 10**4, seed = 0, psa = []):
    
    torch.manual_seed(seed)
    if psa!=[]:
        samples = len(psa)
    
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