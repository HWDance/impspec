import torch
from src.GP_utils import *
from src.kernel_utils import median_heuristic
from src.kernels import NuclearKernel

class baselineGP:
    """
    causalKLGP method for estimating posterior moments of average causal effects
    """

    def __init__(self,Kernel_A, Kernel_V, Kernel_Z, dim_A, dim_V, samples):

        d,p = dim_V, dim_A
        
        # Initialising hypers        
        self.kernel_V = Kernel_V(lengthscale = torch.tensor([d**0.5*1.0]).repeat(d).requires_grad_(True), 
                                scale = torch.tensor([1.0], requires_grad = True))
        self.noise_Y = torch.tensor(-2.0, requires_grad = True).float()
        
        self.kernel_A = Kernel_A(lengthscale = torch.ones(p),
                                  scale = torch.tensor([1.0], requires_grad = True))
        self.noise_feat = torch.tensor(-2.0, requires_grad = True)

    """Will eventually be specific to front and back door"""
    def train(self, Y, A, V, niter, learn_rate, reg = 1e-4, switch_grads_off = True):
    
        """Training P(Y|V)"""
        n,d = V.size()
        Y = Y.reshape(n,)
        
        # Optimiser set up
        params_list = [self.kernel_V.base_kernel.lengthscale,
                                      self.kernel_V.base_kernel.scale,
                                      self.kernel_V.base_kernel.hypers,
                                      self.noise_Y]
        optimizer = torch.optim.Adam(params_list, lr=learn_rate)
        Losses = torch.zeros(niter)
    
        # Updates
        for i in range(niter):
            optimizer.zero_grad()
            loss = -GPML(Y, V, self.kernel_V.base_kernel, self.noise_Y.exp())
            Losses[i] = loss.detach()
            loss.backward()
            optimizer.step()
            if not i % 100:
                    print("iter {0} P(Y|V) loss: ".format(i), Losses[i])
                
        # Disabling gradients
        if switch_grads_off:
            for param in params_list:
                param = param.requires_grad_(False)
            
        """
        Training P(V|A)
        """
        n,p = A.size()

        # Optimiser set up
        params_list = [self.kernel_A.lengthscale,
                       self.kernel_A.hypers,
                      self.kernel_A.scale,
                      self.noise_feat]
        optimizer = torch.optim.Adam(params_list, lr=learn_rate)
        Losses = torch.zeros(niter)
        
        # Updates
        for i in range(niter):
            optimizer.zero_grad()
            loss =  -GPML(V, A, self.kernel_A, self.noise_feat.exp())
            Losses[i] = loss.detach()
            loss.backward()
            optimizer.step()
            if not i % 100:
                print("iter {0} P(V|A) loss: ".format(i), Losses[i])
    
        # Disabling gradients
        if switch_grads_off:
            for param in params_list:
                param = param.requires_grad_(False)
            

    def marginal_post_sample(self,Y,V,A,doA, reg = 1e-4, samples = 10**3, latent = False):

        n, ntest = len(Y), len(doA)
        
        # Base Gaussian samples
        d, p = V.size()[1], A.size()[1]
        epsilon_V = Normal(0,1).sample((ntest, samples, d))
        
        # Moments of p(V|A)
        VdoA_mu = GPpostmean(V, A, doA, self.kernel_A, self.noise_feat.exp(), reg)
        VdoA_var = GPpostvar(V, A, doA, self.kernel_A, self.noise_feat.exp(), reg, latent)

        # Sampling from p(V|A) (marginally)
        VdoA = VdoA_mu[:,None] + VdoA_var.diag().sqrt()[:,None,None]*epsilon_V # ntest x samples x d

        # COME BACK TO DO WITH BATCHING
        YdoA = torch.zeros((ntest, samples))
        for s in range(samples):
            
            # Moments of p(Y|VdoA)
            YdoA_mu = GPpostmean(Y, V, VdoA[:,s], self.kernel_V, self.noise_Y.exp(), reg) # ntest x 1
            YdoA_var = GPpostvar(Y, V, VdoA[:,s], self.kernel_V, self.noise_Y.exp(), reg, latent)  # ntest x ntest
        
            # Sampling from p(Y|VdoA) (marginally)
            epsilon_Y = Normal(0,1).sample((ntest, 1))
            YdoA[:,s] = YdoA_mu + YdoA_var.diag().sqrt()[:,None]*epsilon_Y

        return YdoA




    
