import torch
from src.GP_utils import *
from src.kernel_utils import median_heuristic
from src.kernels import NuclearKernel, ProductKernel
from functools import partial
from copy import deepcopy
import sys

class BayesIMP:
    """
    BayesIMP method for estimating posterior moments of average causal effects
    """

    def __init__(self,Kernel_A, Kernel_V, Kernel_W = None, dim_A = 1, dim_V = 1, dim_W = 1, samples = 10**5, exact = True,
                lengthscale_V_init = 1.0, scale_V_init = 1.0, noise_Y_init = -2.0,
                lengthscale_A_init = 1.0, scale_A_init = 1.0, noise_feat_init = -2.0,
                lengthscale_W_init = 1.0):
        
        d,p,k = dim_V, dim_A, dim_W
        self.exact = exact
        
        # Initialising hypers     
        base_kernel_V = Kernel_V(lengthscale = torch.tensor(d**0.5*lengthscale_V_init).repeat(d).requires_grad_(True), 
                                    scale = torch.tensor(scale_V_init, requires_grad = True))
        self.kernel_V = NuclearKernel(base_kernel_V, 
                                            Normal(torch.zeros(d),torch.ones(1)),
                                            samples = samples)
        if self.exact:
            self.kernel_V.get_gram = self.kernel_V.get_gram_gaussian
        else:
            self.kernel_V.get_gram = self.kernel_V.get_gram_approx
        if Kernel_W != None:
            base_kernel_W = Kernel_W(lengthscale = torch.tensor([k**0.5*lengthscale_W_init]).repeat(k).requires_grad_(True), 
                                scale = torch.tensor([1.0], requires_grad = False))
            self.kernel_W = NuclearKernel(base_kernel_W, 
                                            Normal(torch.zeros(k),torch.ones(1)),
                                            samples = samples)
            if self.exact:
                self.kernel_W.get_gram = self.kernel_W.get_gram_gaussian
            else:
                self.kernel_W.get_gram = self.kernel_W.get_gram_approx
            self.kernel_WV = ProductKernel([self.kernel_W, self.kernel_V])
        
        else:
            self.kernel_W = []
            self.kernel_WV = self.kernel_V
        self.noise_Y = torch.tensor(noise_Y_init, requires_grad = True).float()

        self.kernel_A = Kernel_A(lengthscale = torch.tensor(p**0.5*lengthscale_A_init).repeat(p).requires_grad_(True),
                                  scale = torch.tensor(scale_A_init, requires_grad = True))
        self.noise_feat = torch.tensor(noise_feat_init, requires_grad = True)

    def expand_doA(self, doA, A, intervention_indices):
        """
        Expands doA to match the dimensions of A by repeating doA and interleaving the non-intervention columns from A.
    
        :param doA: Tensor of shape (N, P), where P is the number of intervention indices.
        :param A: Tensor of shape (n0, D), where D is the full dimensionality of the input space.
        :param intervention_indices: List or tensor of indices corresponding to the columns of A that are intervened on.
        :return: Expanded tensor of shape (N * n0, D).
        """
        N, P = doA.shape
        n0, D = A.shape
    
        if len(intervention_indices) != P:
            raise ValueError("The number of intervention_indices must match the second dimension of doA.")
    
        # Create an empty tensor to store the expanded doA
        expanded_doA = torch.empty((N * n0, D))
    
        # Fill in the intervention columns (these are repeated n0 times)
        for i, idx in enumerate(intervention_indices):
            expanded_doA[:, idx] = doA[:, i].repeat(n0)
    
        # Fill in the non-intervention columns (these are interleaved N times)
        average_indices = [j for j in range(D) if j not in intervention_indices]
        for idx in average_indices:
            expanded_doA[:, idx] = A[:, idx].repeat_interleave(N)
    
        return expanded_doA

    """Will eventually be specific to front and back door"""
    def train(self, Y, A, V, W = None, niter = 1000, learn_rate = 0.1, reg = 1e-3, optimise_measure = False, measure_init = 1.0, mc_samples = 100):

        self.kernel_V.samples = mc_samples

        if type(V) != list:
            V = [V,V]
        
        """Training P(Y|V)"""
        n,d = V[1].size()
        Y = Y.reshape(n,)
        
        self.kernel_V.base_kernel.lengthscale = median_heuristic(V[1], per_dimension = True).requires_grad_(True)

        # Optimiser set up
        params_list = [self.kernel_V.base_kernel.lengthscale,
                                      self.kernel_V.base_kernel.scale,
                                      self.kernel_V.base_kernel.hypers,
                                      self.noise_Y]
        self.kernel_V.dist.scale = torch.tensor(measure_init*V[1].var()**0.5).requires_grad_(optimise_measure)
        if optimise_measure:
            params_list.append(self.kernel_V.dist.scale)
            if not self.exact:
                self.kernel_V.get_gram = partial(self.kernel_V.get_gram, rsample = True) 
        # If including W in model (V,W) -> Y, construct product kernel to optimise
        if W != None:
            WV = [W,V[1]]
            self.kernel_W.base_kernel.lengthscale = median_heuristic(W, per_dimension = True).requires_grad_(True)
            params_list.extend([self.kernel_W.base_kernel.lengthscale,
                                      self.kernel_W.base_kernel.scale,
                                      self.kernel_W.base_kernel.hypers]) 
            self.kernel_W.dist.scale = torch.tensor(measure_init*W.var()**0.5).requires_grad_(optimise_measure)
            if optimise_measure:
                params_list.append(self.kernel_W.dist.scale)
                if not self.exact:
                    self.kernel_W.get_gram = partial(self.kernel_W.get_gram, rsample = True) 
        else:
            WV = V[1]
        optimizer = torch.optim.Adam(params_list, lr=learn_rate)
        Losses = torch.zeros(niter)
        
        # Updates
        for i in range(niter):
            optimizer.zero_grad()
            loss = -GPML(Y, WV, self.kernel_WV, torch.exp(self.noise_Y))
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

        self.kernel_A.lengthscale = median_heuristic(A, per_dimension = True).requires_grad_(True)

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
            loss =  -GPfeatureML(V[0], A, self.kernel_V, self.kernel_A, torch.exp(self.noise_feat))
            Losses[i] = loss.detach()
            loss.backward()
            optimizer.step()
            if not i % 100:
                print("iter {0} P(V|A) loss: ".format(i), Losses[i])

        # Disabling gradients
        for param in params_list:
            param = param.requires_grad_(False)

    """Compute E[E[Y|do(A)]] in A -> V -> Y """
    def post_mean(self, Y, A, V, doA, W = None, doW = None, reg=1e-3, samples = 10**5, average_doA = False, intervention_indices=None):
        """
        Compute the posterior mean E[Y|do(A=a)] with selective averaging.
        
        :param Y: Observations from D1, shape (n1,)
        :param A: Observations from D0, shape (n0, D)
        :param V: List of V observations [V0, V1] corresponding to D0 and D1, respectively
        :param doA: Intervention points, shape (N, D)
        :param reg: Regularization parameter
        :param average_doA: Whether to average over the intervention points
        :param intervention_indices: List of column indices of A to intervene on with doA
        :return: Posterior mean, shape (N, 1)
        """
        
        if not self.exact:
            self.kernel_V.samples = samples

        n,N,D = Y.shape[0],doA.shape[0],A.shape[1]
        Y = Y.reshape(n,1)

        # Getting intervention and average indices and constructing expanded doA
        if average_doA:
            average_indices = [j for j in range(D) if j not in intervention_indices]
            expanded_doA = self.expand_doA(doA, A, intervention_indices)  # Shape (n0*N, D)
        else:
            assert doA.shape[1] == A.shape[1]
            expanded_doA = doA  # No expansion if not averaging, expanded_doA: (N, D)

        # One dataset case
        if type(V) != list:

            if W != None:
                WV = [W,V]
                N_w = len(doW)
            else:
                WV = V
                N_w = 1
            
            # Getting kernel matrices
            R_wvwv, R_vv, K_vv, K_aa, k_atest = (
                self.kernel_WV.get_gram(WV, WV),
                self.kernel_V.get_gram(V, V),
                self.kernel_V.get_gram_base(V, V),
                self.kernel_A.get_gram(A, A),
                self.kernel_A.get_gram(expanded_doA, A)
            )
            R_wv = R_wvwv + (self.noise_Y.exp() + reg) * torch.eye(n)
            K_a = K_aa + (self.noise_feat.exp() + reg) * torch.eye(n)

            # Averaging out selected indices
            if average_doA:
                k_atest = k_atest.reshape(n,N,n).mean(0)  
                 
            # Getting components
            E_a = torch.linalg.solve(K_a, k_atest.T)  # (N, n0)
            alpha_y = R_wvwv @ torch.linalg.solve(R_wv, Y)  # (n1, 1)
            if W!=None and doW != None:
                k_wtest = self.kernel_W.get_gram_base(doW,W)
                K_ww = self.kernel_W.get_gram_base(W,W)
            else:
                k_wtest = torch.ones(1,n)
                K_ww = torch.ones((n,n))
            alpha_y = torch.linalg.solve(K_ww*K_vv + reg*torch.eye(n), alpha_y)
            alpha_y = alpha_y*k_wtest.T # (n1, N_w)       
        
            return (E_a.T @ K_vv @ alpha_y).reshape(N*N_w,1)  # (N*N_w, 1) (doW interleaved)
    
        # Two dataset case
        else:
            Vall = torch.row_stack((V[0], V[1]))
            n1, n0 = len(Y), len(A)
    
            # Getting kernel matrices
            R_vv1, R_v1v1, K_vv, K_vv0 = (
                self.kernel_V.get_gram(Vall, V[1]),
                self.kernel_V.get_gram(V[1], V[1]),
                self.kernel_V.get_gram_base(Vall, Vall),
                self.kernel_V.get_gram_base(Vall, V[0])
            )
            K_aa, k_atest = (
                self.kernel_A.get_gram(A, A),
                self.kernel_A.get_gram(expanded_doA, A)
            )
            R_v1 = R_v1v1 + (self.noise_Y.exp() + reg) * torch.eye(n1)
            K_a = K_aa + (self.noise_feat.exp() + reg) * torch.eye(n0)

            # Averaging out selected indices
            if average_doA:
                k_atest = k_atest.reshape(n0,N,n0).mean(0)  
                 
            # Getting components
            E_a = torch.linalg.solve(K_a, k_atest.T)  # (n0, N)
            alpha_y = torch.linalg.solve(R_v1, Y)  # (n1, 1)
    
            return E_a.T @ K_vv0.T @ torch.linalg.solve(K_vv + torch.eye(n1+n0)*reg, R_vv1) @ alpha_y  # (N, 1)
            
    """Compute Var[E[Y|do(A)]] in A -> V -> Y"""
    def post_var(self, Y, A, V, doA, doA2 = None, W = None, doW = None, doW2 = None,
                 reg=1e-3, latent=True, samples=10**5, average_doA=False, intervention_indices=None, diag = True):
        
        if not self.exact:
            self.kernel_V.samples = samples
    
        # Getting second set of doA and doW if computing covariance
        if doA2 == None:
            doA2 = doA
        if doW2 == None and doW != None:
            doW2 = doW
        
        # Dimensions
        n = len(Y)
        n0, N, M, D = len(A), len(doA), len(doA2), A.shape[1]
        Y = Y.reshape(n, 1)
            
        # Expand doA with replacement using the selected columns from A
        if average_doA:
            average_indices = [j for j in range(D) if j not in intervention_indices]
            expanded_doA = self.expand_doA(doA, A, intervention_indices)  # Shape (n0*N, D)
            expanded_doA2 = self.expand_doA(doA2, A, intervention_indices)  # Shape (n0*M, D)
        else:
            assert doA.shape[1] == A.shape[1]
            assert doA2.shape[1] == A.shape[1]
            expanded_doA = doA  # No expansion if not averaging, expanded_doA: (N, D)
            expanded_doA2 = doA2  # No expansion if not averaging, expanded_doA: (M, D)
        
        # One dataset case
        if type(V) != list:

            if W != None:
                WV = [W,V]
                N_w = len(doW)
                M_w = len(doW2)
            else:
                WV = V
                N_w = 1
                M_w = 1
            
            # Getting kernel matrices
            R_wvwv,R_vv, K_vv, K_aa, k_atest, k_atest2 = (
                self.kernel_WV.get_gram(WV, WV),
                self.kernel_V.get_gram(V, V),
                self.kernel_V.get_gram_base(V, V),
                self.kernel_A.get_gram(A, A),
                self.kernel_A.get_gram(expanded_doA, A),
                self.kernel_A.get_gram(expanded_doA2, A)
            )
            R_wv = R_wvwv + (self.noise_Y.exp() + reg) * torch.eye(n)
            K_a = K_aa + (self.noise_feat.exp() + reg) * torch.eye(n)
            R_wvwv_bar = R_wvwv - R_wvwv @ torch.linalg.solve(R_wv, R_wvwv)

            if W != None and doW!= None:
                K_wtestwtest = self.kernel_W.get_gram_base(doW,doW2) # (N_w, M_w)
                k_wtest = self.kernel_W.get_gram_base(doW,W) # (N_w, n1)
                k_wtest2 = self.kernel_W.get_gram_base(doW2,W) # (M_w, n1)
                K_ww = self.kernel_W.get_gram_base(W,W)
            else:
                K_wtestwtest = torch.ones((1,1))
                k_wtest = torch.ones((1,n))
                k_wtest2 = torch.ones((1,n))
                K_ww = torch.ones((n,n))
                
            # Averaging out selected indices
            if average_doA:
                k_atest = k_atest.reshape(n0,N,n0).mean(0) # (N,n0)
                k_atest2 = k_atest2.reshape(n0,M,n0).mean(0) # (M,n0)            
            
            # Computing matrix vector products
            beta_a = torch.linalg.solve(K_a, k_atest.T)  # (n0, N)
            beta_a2 = torch.linalg.solve(K_a, k_atest2.T)  # (n0, N)
            alpha_y = torch.linalg.solve(R_wv, Y)  # (n1, 1)
            KinvR = torch.linalg.solve(K_vv*K_ww + torch.eye(n) * reg, R_wvwv)  # (n1, n1)
            alpha = KinvR @ alpha_y
          
            # Get gram matrix on doA,doA2
            if average_doA: 
                
                # If averaging, define separate kernels for averaging and intervention indices
                kernel_Aavg = deepcopy(self.kernel_A)
                kernel_Aavg.lengthscale = self.kernel_A.lengthscale[average_indices]
                kernel_Aavg.scale = 1.0
    
                kernel_doA = deepcopy(self.kernel_A)
                kernel_doA.lengthscale = self.kernel_A.lengthscale[intervention_indices]
                
                # Construct average and interventional gram matrices
                Aavg = A[:,average_indices].reshape(n0,len(average_indices))
                K_Aavg = kernel_Aavg.get_gram(Aavg,Aavg).mean()
                K_doA = kernel_doA.get_gram(doA,doA2)
    
                K_atestatest = K_Aavg*K_doA
            else:
                # otherwise, just construct N x M gram matrix on doA 
                K_atestatest = self.kernel_A.get_gram(doA,doA2)
            
            # Final computations
            if not diag:
                kpost_atest = K_atestatest - k_atest @ torch.linalg.solve(K_a, k_atest2.T) # (N,M)                
                # If N larger return N_w lots of N x M post cov
                if N >= N_w:
                    posterior_variance = torch.zeros((N,M,N_w))

                    for w in range(N_w):
    
                        gamma_w = torch.linalg.solve(K_ww*K_vv + reg*torch.eye(n), k_wtest[w].diag() @ K_vv @ beta_a)
                        gamma_w2 = torch.linalg.solve(K_ww*K_vv + reg*torch.eye(n), k_wtest2[w].diag() @ K_vv @ beta_a2)
                        V1 = gamma_w.T @ R_wvwv_bar @ gamma_w2  # (N,M)

                        V2 = kpost_atest * (alpha.T @ k_wtest[w].diag() @ R_vv @ k_wtest2[w].diag() @ alpha).view(-1)

                        KinvDwRDw = torch.linalg.solve(K_vv*K_ww + torch.eye(n)*reg, k_wtest[w].diag() @ R_vv @ k_wtest2[w].diag())
                        V3 = kpost_atest * torch.trace(torch.linalg.solve(K_vv*K_ww + torch.eye(n) * reg, R_wvwv_bar @ KinvDwRDw))  # (N,M)
                        
                        posterior_variance[...,w] = V1 + V2 + V3  # (N,M)
             
                # otherwise return N lots of N_w x M_w post cov   
                else:
                    posterior_variance = torch.zeros((N_w, M_w, N))

                    for a in range(N):
                        gamma_a = torch.linalg.solve(K_vv*K_ww + reg*torch.eye(n), (K_vv @ beta_a[:,a].reshape(n,1)) * k_wtest.T)
                        gamma_a2 = torch.linalg.solve(K_vv*K_ww + reg*torch.eye(n), (K_vv @ beta_a2[:,a].reshape(n,1)) * k_wtest2.T)
                        V1 = gamma_a.T @ R_wvwv_bar @ gamma_a2

                        V2 = kpost_atest[a,a] * (k_wtest.T * alpha).T @ R_vv @ (k_wtest2.T * alpha) 

                        KinvRbarKinv = torch.linalg.solve(K_vv*K_ww + torch.eye(n)*reg, R_wvwv_bar) @ torch.linalg.inv(K_vv*K_ww + torch.eye(n)*reg)
                        V3 = kpost_atest[a,a] * k_wtest @ (R_vv * KinvRbarKinv) @ k_wtest2.T

                        posterior_variance[...,a] = V1 + V2 + V3 # (N_w, M_w)
                
                # If only one value for doA or doW, remove extra dimension to return (N, M) or (N_w, M_w)
                if N == 1 or N_w == 1:
                    posterior_variance = posterior_variance[...,0]  
                    
            # Or if diag just directly return (N x N_w,) diagonal                       
            else:
                kpost_atest = K_atestatest.diag() - (torch.linalg.solve(K_a, k_atest2.T)*k_atest.T).sum(0) # (N,)

                gamma_aw = k_wtest.T.repeat_interleave(N,dim = 1) * (K_vv @ beta_a).repeat(1,N_w) # (n1, N*N_w)
                gamma_aw2 = k_wtest2.T.repeat_interleave(M,dim = 1) * (K_vv @ beta_a2).repeat(1,M_w) # (n1, N*N_w)
                gamma_aw = torch.linalg.solve(K_vv*K_ww + reg*torch.eye(n),gamma_aw)
                gamma_aw2 = torch.linalg.solve(K_vv*K_ww + reg*torch.eye(n),gamma_aw2)
                V1 = ((R_wvwv_bar @ gamma_aw2)*gamma_aw2).sum(0)  # (N*N_w,)
                
                A_w = ((k_wtest.T * alpha).T @ R_vv @ (k_wtest2.T * alpha))
                V2 = torch.kron(kpost_atest,A_w.diag())
                
                # (N,)
                KinvRbarKinv = torch.linalg.solve(K_vv*K_ww + torch.eye(n)*reg, R_wvwv_bar) @ torch.linalg.inv(K_vv*K_ww + torch.eye(n)*reg)
                B_w = k_wtest @ (R_vv * KinvRbarKinv) @ k_wtest2.T
                V3 = torch.kron(kpost_atest, B_w.diag())# (N,)
                                
                posterior_variance = (V1 + V2 + V3).reshape(N*N_w,1)  # (N,1)

            return posterior_variance  

        # Split datasets case
        else:
            Vall = torch.row_stack((V[0], V[1]))
            n1, n0 = len(Y), len(A)
            
            # Getting kernel matrices
            R_v1v0, R_v0v0, R_v1v1, R_vv1, R_vv0, R_vv = (
                self.kernel_V.get_gram(V[1], V[0]),
                self.kernel_V.get_gram(V[0], V[0]),
                self.kernel_V.get_gram(V[1], V[1]),
                self.kernel_V.get_gram(Vall, V[1]),
                self.kernel_V.get_gram(Vall, V[0]),
                self.kernel_V.get_gram(Vall, Vall)
            )
            K_v0v0, K_v1v1, K_vv1, Kvv0, K_vv = (
                self.kernel_V.get_gram_base(V[0], V[0]),
                self.kernel_V.get_gram_base(V[1], V[1]),
                self.kernel_V.get_gram_base(Vall, V[1]),
                self.kernel_V.get_gram_base(Vall, V[0]),
                self.kernel_V.get_gram_base(Vall, Vall)
            )
            K_aa, k_atest,k_atest2 = (
                self.kernel_A.get_gram(A, A),
                self.kernel_A.get_gram(expanded_doA, A),
                self.kernel_A.get_gram(expanded_doA2, A)
            )
            R_v1 = R_v1v1 + (self.noise_Y.exp() + reg) * torch.eye(n1)
            K_v1 = K_v1v1 + (self.noise_Y.exp() + reg) * torch.eye(n1)
            K_a = K_aa + (self.noise_feat.exp() + reg) * torch.eye(n0)
            
            # Averaging out selected indices
            if average_doA:
                k_atest = k_atest.reshape(n0,N,n0).mean(0) # (N,n0)
                k_atest2 = k_atest2.reshape(n0,M,n0).mean(0) # (M,n0)    

            # Computing matrix vector products
            Theta1 = torch.linalg.solve(K_vv+torch.eye(n1+n0)*reg, R_vv0) @ torch.linalg.solve(R_v0v0+torch.eye(n0)*reg, K_v0v0)  # (n1+n0, n1+n0)
            Theta4 = torch.linalg.solve(K_vv+torch.eye(n1+n0)*reg, R_vv1) @ torch.linalg.solve(R_v1, Y)  # (n1+n0, 1)
            Theta2a = Theta4.T @ R_vv @ Theta4  # (1, 1)
            Theta2b = Theta4.T @ R_vv0 @ torch.linalg.solve(R_v0v0+torch.eye(n0)*reg, R_vv0.T) @ Theta4  # (1, 1)
            Theta3a = torch.trace(torch.linalg.solve(K_vv+torch.eye(n1+n0)*reg, R_vv) @ torch.linalg.solve(K_vv+torch.eye(n1+n0)*reg, R_vv - R_vv1 @ torch.linalg.solve(R_v1, R_vv1.T)))  # scalar
            Theta3b = torch.trace(torch.linalg.solve(K_vv+torch.eye(n1+n0)*reg, R_vv0) 
                                  @ torch.linalg.solve(R_v0v0+torch.eye(n0)*reg, R_vv0.T) 
                                  @ torch.linalg.solve(K_vv+torch.eye(n1+n0)*reg, R_vv - R_vv1 @ torch.linalg.solve(R_v1, R_vv1.T)))  # scalar 
            E_a = torch.linalg.solve(K_a, k_atest.T)  # (n0, N)
            E_a2 = torch.linalg.solve(K_a, k_atest2.T)  # (n0, M)
            G_aa = E_a.T @ k_atest2.T

            # Get gram matrix on doA,doA2
            if average_doA: 
                
                # If averaging, define separate kernels for averaging and intervention indices
                kernel_Aavg = deepcopy(self.kernel_A)
                kernel_Aavg.lengthscale = self.kernel_A.lengthscale[average_indices]
                kernel_Aavg.scale = 1.0
    
                kernel_doA = deepcopy(self.kernel_A)
                kernel_doA.lengthscale = self.kernel_A.lengthscale[intervention_indices]
                
                # Construct average and interventional gram matrices
                Aavg = A[:,average_indices].reshape(n0,len(average_indices))
                K_Aavg = kernel_Aavg.get_gram(Aavg,Aavg).mean()
                K_doA = kernel_doA.get_gram(doA,doA2)
    
                K_atestatest = K_Aavg*K_doA
            else:
                # otherwise, just construct N x M gram matrix on doA 
                K_atestatest = self.kernel_A.get_gram(doA,doA2)            
            F_aa = K_atestatest  # (N,M)
            
            # Final computations
            V1 = E_a.T @ Theta1.T @ (R_vv - R_vv1 @ torch.linalg.solve(R_v1, R_vv1.T)) @ Theta1 @ E_a2  # (N,M)
            V2 = Theta2a * F_aa - Theta2b * G_aa  # (N,M)
            V3 = Theta3a * F_aa - Theta3b * G_aa  # (N,M)

            if not diag:
                posterior_variance = V1 + V2 + V3 # (N,M)
            else:
                posterior_variance = (V1 + V2 + V3).diag().reshape(N,1)  # (N,1)
    
            return posterior_variance
    

        
            
            