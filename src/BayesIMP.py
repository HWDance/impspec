import torch
from src.GP_utils import *
from src.kernel_utils import median_heuristic
from src.kernels import NuclearKernel
from functools import partial

class BayesIMP:
    """
    BayesIMP method for estimating posterior moments of average causal effects
    """

    def __init__(self,Kernel_A, Kernel_V, Kernel_Z, dim_A, dim_V, samples, exact):
        d,p = dim_V, dim_A
        self.exact = exact
        
        # Initialising hypers     
        base_kernel_V = Kernel_V(lengthscale = torch.tensor([d**0.5*1.0]).repeat(d).requires_grad_(True), 
                                    scale = torch.tensor([1.0], requires_grad = True))
        self.kernel_V = NuclearKernel(base_kernel_V, 
                                            Normal(torch.zeros(d),torch.ones(1)),
                                            samples = samples)
        if self.exact:
            self.kernel_V.get_gram = self.kernel_V.get_gram_gaussian
        else:
            self.kernel_V.get_gram = self.kernel_V.get_gram_approx
        self.noise_Y = torch.tensor(-2.0, requires_grad = True).float()

        self.kernel_A = Kernel_A(lengthscale = torch.ones(p).requires_grad_(True),
                                  scale = torch.tensor([1.0], requires_grad = True))
        self.noise_feat = torch.tensor(-2.0, requires_grad = True)

    """Will eventually be specific to front and back door"""
    def train(self, Y, A, V, niter, learn_rate, reg = 1e-4, optimise_measure = False, measure_init = 1.0, mc_samples = 10):

        self.kernel_V.samples = mc_samples
        
        """Training P(Y|V)"""
        n,d = V.size()
        Y = Y.reshape(n,)

        
        # Optimiser set up
        params_list = [self.kernel_V.base_kernel.lengthscale,
                                      self.kernel_V.base_kernel.scale,
                                      self.kernel_V.base_kernel.hypers,
                                      self.noise_Y]
        self.kernel_V.dist.scale = torch.tensor(measure_init*V.var()**0.5).requires_grad_(optimise_measure)
        if optimise_measure:
            params_list.append(self.kernel_V.dist.scale)
            if not self.exact:
                self.kernel_V.get_gram = partial(self.kernel_V.get_gram, rsample = True)
            
        optimizer = torch.optim.Adam(params_list, lr=learn_rate)
        Losses = torch.zeros(niter)
    
        # Updates
        for i in range(niter):
            optimizer.zero_grad()
            loss = -GPML(Y, V, self.kernel_V, torch.exp(self.noise_Y))
            Losses[i] = loss.detach()
            loss.backward()
            optimizer.step()
            if not i % 100:
                    print("iter {0} P(Y|V) loss: ".format(i), Losses[i])

        # Disabling gradients
        for param in params_list:
            param = param.requires_grad_(False)
            
        """
        Training P(V|A)
        """
        n,p = A.size()

        # Getting lengthscale
        #self.kernel_A.lengthscale = median_heuristic(A)
        
        # Optimiser set up
        params_list = [self.kernel_A.hypers,
                       self.kernel_A.lengthscale,
                        self.kernel_A.scale,
                        self.noise_feat]
        optimizer = torch.optim.Adam(params_list, lr=learn_rate)
        Losses = torch.zeros(niter)
        
        # Updates
        for i in range(niter):
            optimizer.zero_grad()
            loss =  -GPfeatureML(V, A, self.kernel_V, self.kernel_A, torch.exp(self.noise_feat))
            Losses[i] = loss.detach()
            loss.backward()
            optimizer.step()
            if not i % 100:
                print("iter {0} P(V|A) loss: ".format(i), Losses[i])

        # Disabling gradients
        for param in params_list:
            param = param.requires_grad_(False)

    """Compute E[E[Y|do(A)]] in A -> V -> Y """
    def post_mean(self, Y, A, V, doA, reg = 1e-4, samples = 10**3):
        if not self.exact:
            self.kernel_V.samples = samples

        n = len(Y)
        Y = Y.reshape(n,1)
        
        # getting kernel matrices
        R_vv,K_aa,k_atest = (self.kernel_V.get_gram(V,V),
                             self.kernel_A.get_gram(A,A),
                             self.kernel_A.get_gram(doA, A))
        R_v = R_vv+(self.noise_Y.exp()+reg)*torch.eye(n)
        K_a = K_aa+(self.noise_feat.exp()+reg)*torch.eye(n)

        # Getting components
        A_a = torch.linalg.solve(K_a,k_atest.T).T
        alpha_y = torch.linalg.solve(R_v,Y)
        
        return  A_a @ R_vv @ alpha_y
        
    """Compute Var[E[Y|do(A)]] in A -> V -> Y """
    def post_var(self, Y, A, V, doA, reg = 1e-4, latent = True, samples = 10**3):
        if not self.exact:
            self.kernel_V.samples = samples

        n = len(Y)
        Y = Y.reshape(n,1)
        
        # getting kernel matrices
        R_vv,K_vv,K_aa,k_atest, k_atestatest = (self.kernel_V.get_gram(V,V),
                                 self.kernel_V.get_gram_base(V,V),
                                 self.kernel_A.get_gram(A,A),
                                 self.kernel_A.get_gram(doA, A),
                                 self.kernel_A.get_gram(doA, doA))
        R_v = R_vv+(self.noise_Y.exp()+reg)*torch.eye(n)
        K_v = K_vv+(self.noise_Y.exp()+reg)*torch.eye(n)
        K_a = K_aa+(self.noise_feat.exp()+reg)*torch.eye(n)
        R_vv_bar = R_vv - R_vv @ torch.linalg.solve(R_v,R_vv)+ (not latent)*self.noise_Y.exp()*torch.eye(n)
        kpost_atest_approx = (k_atestatest - k_atest @ torch.linalg.solve(K_a,k_atest.T))+ (not latent)*self.noise_feat.exp()*torch.eye(len(doA))
        
        # computing matrix vector products
        alpha_a = torch.linalg.solve(K_a,k_atest.T)
        alpha_y = torch.linalg.solve(K_v,Y)
        KinvR = torch.linalg.solve(K_vv+torch.eye(n)*reg,R_vv)
        
        V1 = alpha_a.T @ R_vv_bar @ alpha_a
        V2 = alpha_y.T @ R_vv @ KinvR @ KinvR @ alpha_y * kpost_atest_approx
        V3 = torch.trace(torch.linalg.solve(K_vv+torch.eye(n)*reg, R_vv_bar @ KinvR)) * kpost_atest_approx
        
        return V1+V2+V3