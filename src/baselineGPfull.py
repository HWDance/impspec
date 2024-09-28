import torch
from src.GP_utils import *
from src.kernel_utils import median_heuristic
from src.kernels import NuclearKernel, ProductKernel

class baselineGP:
    """
    causalKLGP method for estimating posterior moments of average causal effects
    """

    def __init__(self,Kernel_A, Kernel_V, Kernel_W = None, dim_A = 1, dim_V = 1, dim_W = 1, single_kernel = False,
                lengthscale_V_init = 1.0, scale_V_init = 1.0, noise_Y_init = -2.0,
                lengthscale_A_init = 1.0, scale_A_init = 1.0, noise_feat_init = -2.0,
                lengthscale_W_init = 1.0):

        d,p,k = dim_V, dim_A, dim_W

        self.single_kernel = single_kernel

        self.kernel_V = Kernel_V(lengthscale = torch.tensor([d**0.5*lengthscale_V_init]).repeat(d).requires_grad_(True), 
                                    scale = torch.tensor(scale_V_init, requires_grad = True))

        if Kernel_W != None:
            self.kernel_W =  Kernel_W(lengthscale = torch.tensor([k**0.5*lengthscale_W_init]).repeat(k).requires_grad_(True), 
                                scale = torch.tensor([1.0], requires_grad = False))
            self.kernel_WV = ProductKernel([self.kernel_W, self.kernel_V])
        else:
            self.kernel_W = []
            self.kernel_WV = self.kernel_V

        self.noise_Y = torch.tensor(noise_Y_init, requires_grad = True).float()

        self.kernel_A = []
        self.noise_feat = []
        
        # Initialising hypers  
        if self.single_kernel:
            kernel_a = Kernel_A(lengthscale = torch.tensor(p**0.5*lengthscale_A_init).repeat(p).requires_grad_(True),
                                  scale = torch.tensor(scale_A_init, requires_grad = True))
            noise = (noise_feat_init*torch.ones(1)).requires_grad_(True)
            self.kernel_A.extend([kernel_a]*d)
            self.noise_feat.extend([noise]*d)  
        else:
            for j in range(d):
                self.kernel_A.append(Kernel_A(lengthscale = torch.tensor(p**0.5*lengthscale_A_init).repeat(p).requires_grad_(True),
                                      scale = torch.tensor(scale_A_init, requires_grad = True)))
                self.noise_feat.append((noise_feat_init*torch.ones(1)).requires_grad_(True))
            

    """Will eventually be specific to front and back door"""
    def train(self, Y, A, V, W= None, niter = 500, learn_rate = 0.1, reg = 1e-4, switch_grads_off = True, force_PD = False, median_heuristic_A = False):
    
        # Constructing list of V_1,V_2 for compatibility with data fusion case
        if type(V) != list:
            V = [V,V]
        
        """Training P(Y|V)"""
        n,d = V[0].size()
        Y = Y.reshape(n,)
        
        # Optimiser set up
        params_list = [self.kernel_V.lengthscale,
                                      self.kernel_V.scale,
                                      self.kernel_V.hypers,
                                      self.noise_Y]
        # If including W in model (V,W) -> Y, construct product kernel to optimise
        if W != None:
            WV = [W,V[1]]
            params_list.extend([self.kernel_W.lengthscale,
                                      self.kernel_W.scale,
                                      self.kernel_W.hypers])
        else:
            WV = V[1]
        optimizer = torch.optim.Adam(params_list, lr=learn_rate)
        Losses = torch.zeros(niter)
            
        # Updates
        for i in range(niter):
            optimizer.zero_grad()
            loss = -GPML(Y, WV, self.kernel_WV, self.noise_Y.exp(), force_PD = force_PD)
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
        params_list = []
        for j in range(d):
            params_list.extend([self.kernel_A[j].hypers,
                      self.kernel_A[j].scale,
                      self.noise_feat[j]])
            if median_heuristic_A:
                self.kernel_A[j].lengthscale = median_heuristic(A)
            else:
                params_list.append(self.kernel_A[j].lengthscale)
        optimizer = torch.optim.Adam(params_list, lr=learn_rate)
        Losses = torch.zeros(niter)
        
        # Updates
        for i in range(niter):
            optimizer.zero_grad()
            loss = 0
            for j in range(d):
                loss +=  -GPML(V[0][:,j], A, self.kernel_A[j], self.noise_feat[j].exp(), force_PD = force_PD)
            Losses[i] = loss.detach()
            loss.backward()
            optimizer.step()
            if not i % 100:
                print("iter {0} P(V|A) loss: ".format(i), Losses[i])
    
        # Disabling gradients
        if switch_grads_off:
            for param in params_list:
                param = param.requires_grad_(False)
            

    def marginal_post_sample(self,Y,V,A,doA, W=None, doW = None, reg = 1e-3, error_samples = 10**2, gp_samples = 10**2):

        # Constructing list of V_1,V_2 for compatibility with data fusion case
        if type(V) != list:
            V = [V,V]
        
        n, ntest = len(Y), len(doA)
        d, p = V[0].size()[1], A.size()[1]
        Y = Y.reshape(n,1)
        if W != None:
            WV = [W,V[1]]
            nwtest = len(doW)
        else:
            WV = V[1]
            nwtest = 1

        # Getting kernel matrices
        K_wv = self.kernel_WV.get_gram(WV,WV)+torch.eye(n)*self.noise_Y.exp()
        if W != None and doW!= None:
            k_wtest = self.kernel_W.get_gram(doW,W)
            K_wtestwtest = self.kernel_W.get_gram(doW,doW)
        else:
            k_wtest = torch.ones(1,n)
            K_wtestwtest = torch.eye(1)
            
        
        # Base Gaussian samples
        U_v1 = Normal(0,1).sample((ntest, error_samples, d))*torch.tensor(self.noise_feat).exp()**0.5
        U_v2 = Normal(0,1).sample((ntest, error_samples, d))*torch.tensor(self.noise_feat).exp()**0.5
        U_y = Normal(0,1).sample((ntest, 1))*self.noise_Y.exp()**0.5
        
        # Posterior moments of E[V|A]
        VdoA_mu = torch.zeros((ntest,d))
        VdoA_var = torch.zeros((ntest,ntest,d))
        for j in range(d):
            VdoA_mu[...,j] =  GPpostmean(V[0][:,j], A, doA, self.kernel_A[j], self.noise_feat[j].exp(), reg) # ntest x 1 (d rows of this)
            VdoA_var[...,j] = GPpostvar(A, doA, self.kernel_A[j], self.noise_feat[j].exp(), reg, latent = True) # ntest x ntest (d rows of this)

        # Sampling from E[V|A] (marginally)
        EVdoA_samples = torch.zeros((ntest,gp_samples,d))
        for j in range(d):
            epsilon_V = Normal(0,1).sample((ntest, gp_samples)) 
            EVdoA_samples[...,j] = VdoA_mu[:,j:j+1] + VdoA_var[...,j].diag().sqrt()[:,None]*epsilon_V  # ntest x gp_samples (d rows of this)

        EYdoA_samples = torch.zeros((ntest, gp_samples, nwtest))

        # Now iteratively get moments of E[Y|do(a), f, g] | g  ~ N(m_g(a), v_g(a))
        K_v =self.kernel_V.get_gram(V[1],V[1])+torch.eye(n)*self.noise_Y.exp()
        for i in range(ntest):
            
            # (i) Computing E[k(g(a)+e, g'(a) + e')|g=g'= hat g]
            g_a = EVdoA_samples[i] # gp_samples x d
            u_v1 = U_v1[i] # error_samples x d
            u_v2 = U_v2[i] # error_samples x d

            vdoa1 = g_a[:,None] + u_v1[None] # gp_samples x error_samples x d
            vdoa2 = g_a[:,None] + u_v2[None] # gp_samples x error_samples x d

            vdoa1 = vdoa1.reshape(gp_samples*error_samples, 1, d)
            vdoa2 = vdoa2.reshape(gp_samples*error_samples, 1, d)

            Ek_vdoa1_vdoa2 = self.kernel_V.get_gram(vdoa1,vdoa2).reshape(gp_samples, error_samples).mean(1) # gp_samples x 0

            # (ii) Computing E[k(g(a)+e,V_tr)]
            vdoa1 = vdoa1[:,0] # gp_samples*error_samples x d
            Ek_vdoa1_V = self.kernel_V.get_gram(vdoa1, V[1]) # gp_samples*error_samples x n
            Ek_vdoa1_V = Ek_vdoa1_V.reshape(gp_samples, error_samples, n).mean(1) # gp_samples x n

            # Computing moments

            for w in range(nwtest):
                m_ga = Ek_vdoa1_V*k_wtest[w].reshape(1,n) @  torch.linalg.solve(K_wv,Y) # gp_samples x 1
                
                v_ga = (Ek_vdoa1_vdoa2*K_wtestwtest[w,w] - 
                        ((Ek_vdoa1_V*k_wtest[w].reshape(1,n))  @ 
                         torch.linalg.solve(K_v, Ek_vdoa1_V.T*k_wtest[w].reshape(n,1))).diag()).abs() # gp_samples x 1
                
                # Sampling
                epsilon_Y = Normal(0,1).sample((gp_samples,))
                
                EYdoA_samples[i,...,w] = m_ga[:,0] + v_ga**0.5*epsilon_Y
                
        #Removing unnecessary dimensions
        if nwtest == 1:
            EYdoA_samples = EYdoA_samples[...,0]
        if ntest == 1:
            EYdoA_samples = EYdoA_samples[0].T
            
        return EYdoA_samples, EVdoA_samples




    
