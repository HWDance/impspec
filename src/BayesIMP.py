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

    def expand_doA_with_replacement(doA, A, average_indices):
        """
        Expand doA by selectively replacing specified columns with those from A, and
        perform a Kronecker-like product expansion for averaging.
        
        :param doA: Original intervention points, shape (N, D)
        :param A: Observations from D0, shape (n0, D)
        :param average_indices: List of column indices to replace and average over
        :return: Expanded doA, shape (n0*N, D)
        """
        N, D = doA.shape
        n0, _ = A.shape
        
        # Initialize the expanded doA
        expanded_doA = doA.repeat(n0, 1)  # Shape (n0*N, D)
    
        # Replace the selected columns in expanded doA with corresponding columns from A
        for idx in average_indices:
            expanded_doA[:, idx] = A[:, idx].repeat_interleave(N)

        return expanded_doA

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
    def post_mean(self, Y, A, V, doA, reg = 1e-4, samples = 10**5, average_doA = False, average_indices=None):
        
        if not self.exact:
            self.kernel_V.samples = samples

        n,N = len(Y),len(doA)
        Y = Y.reshape(n,1)

        if average_doA and average_indices is not None:
            expanded_doA = self.expand_doA_with_replacement(doA, A, average_indices)  # (n0*N, D)
        else:
            expanded_doA = doA

        # One dataset case
        if type(V) != list:
            # Getting kernel matrices
            R_vv, K_aa, k_atest = (
                self.kernel_V.get_gram(V, V),
                self.kernel_A.get_gram(A, A),
                self.kernel_A.get_gram(expanded_doA, A)
            )
            R_v = R_vv + (self.noise_Y.exp() + reg) * torch.eye(n)
            K_a = K_aa + (self.noise_feat.exp() + reg) * torch.eye(n)
    
            # Getting components
            A_a = torch.linalg.solve(K_a, k_atest.T).T  # (n0*N, n0)
            alpha_y = torch.linalg.solve(R_v, Y)  # (n, 1)
    
            post_mean = A_a @ R_vv @ alpha_y  # (n0*N, 1)
    
            if average_doA:
                post_mean = post_mean.reshape(n, N).mean(0).reshape(N, 1)  # (N, 1)
            return post_mean
    
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
    
            # Getting components
            E_a = torch.linalg.solve(K_a, k_atest.T)  # (n0, n0 x N)
            alpha_y = torch.linalg.solve(R_v1, Y)  # (n1, 1)
    
            post_mean = E_a.T @ K_vv0.T @ torch.linalg.solve(K_vv, R_vv1) @ alpha_y  # (n0*N, 1)
    
            if average_doA:
                post_mean = post_mean.reshape(n0, N).mean(0).reshape(N, 1)  # (N, 1)
            return post_mean
        
    """Compute Var[E[Y|do(A)]] in A -> V -> Y"""
    def post_var(self, Y, A, V, doA, reg=1e-4, latent=True, samples=10**5, average_doA=False, average_indices=None):
        
        if not self.exact:
            self.kernel_V.samples = samples
    
        n = len(Y)
        n0, N = len(A), len(doA)
        Y = Y.reshape(n, 1)
    
        if average_doA and average_indices is not None:
            expanded_doA = self.expand_doA_with_replacement(doA, A, average_indices)  # (n0*N, D)
        else:
            expanded_doA = doA
        
        # One dataset case
        if type(V) != list:
            # Getting kernel matrices
            R_vv, K_vv, K_aa, k_atest = (
                self.kernel_V.get_gram(V, V),
                self.kernel_V.get_gram_base(V, V),
                self.kernel_A.get_gram(A, A),
                self.kernel_A.get_gram(expanded_doA, A)
            )
            
            R_v = R_vv + (self.noise_Y.exp() + reg) * torch.eye(n)
            K_v = K_vv + (self.noise_Y.exp() + reg) * torch.eye(n)
            K_a = K_aa + (self.noise_feat.exp() + reg) * torch.eye(n)
            R_vv_bar = R_vv - R_vv @ torch.linalg.solve(R_v, R_vv)
    
            # Efficient computation of kpost_atest using the .sum(0) trick
            kpost_atest = GPpostvar(
                X=A,  # Training data (D0)
                Xtest=expanded_doA,  # Test data (expanded intervention points)
                kernel=self.kernel_A,
                noise=torch.exp(self.noise_feat),  # Noise term for the kernel
                reg=reg,
                latent=latent,
                diag=True  # Use the diagonal-only computation for efficiency
                )  # kpost_atest: (n0*N,) if expanded, or (N,) otherwise
    
            # Computing matrix vector products
            alpha_a = torch.linalg.solve(K_a, k_atest.T)  # (n0, n0*N)
            alpha_y = torch.linalg.solve(K_v, Y)  # (n, 1)
            KinvR = torch.linalg.solve(K_vv + torch.eye(n) * reg, R_vv)  # (n, n)
    
            V1 =  ((R_vv_bar @ alpha_a)*alpha_a).sum(0)  # (n0*N,)
            V2 = kpost_atest * (alpha_y.T @ R_vv @ KinvR @ KinvR @ alpha_y).view(-1)  # (n0*N,)
            V3 = kpost_atest * torch.trace(torch.linalg.solve(K_vv + torch.eye(n) * reg, R_vv_bar @ KinvR))  # (n0*N,)
    
            posterior_variance = V1 + V2 + V3  # (n0*N,)
    
            if average_doA:
                posterior_variance = posterior_variance.reshape(n0, N).mean(0).reshape(N, 1)  # (N, 1)
            else:
                posterior_variance = posterior_variance.reshape(N, 1)  # (N, 1)
                
            return posterior_variance  

        # Two datasets case
        else:
            Vall = torch.row_stack((V[0], V[1]))
            n1, n0 = len(Y), len(A)
            
            # Getting kernel matrices
            R_v1v0, R_v0v0, R_v1v1, R_vv1, Rvv0, R_vv = (
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
            
            K_aa, k_atest = (
                self.kernel_A.get_gram(A, A),
                self.kernel_A.get_gram(expanded_doA, A)
            )
    
            R_v1 = R_v1v1 + (self.noise_Y.exp() + reg) * torch.eye(n1)
            K_v1 = K_v1v1 + (self.noise_Y.exp() + reg) * torch.eye(n1)
            K_a = K_aa + (self.noise_feat.exp() + reg) * torch.eye(n0)
            
            # Efficient computation of kpost_atest using the .sum(0) trick
            kpost_atest = GPpostvar(
                X=A,  # Training data (D0)
                Xtest=expanded_doA,  # Test data (expanded intervention points)
                kernel=self.kernel_A,
                noise=torch.exp(self.noise_feat),  # Noise term for the kernel
                reg=reg,
                latent=latent,
                diag=True  # Use the diagonal-only computation for efficiency
                )  # kpost_atest: (n0*N,) if expanded, or (N,) otherwise
    
            E_a = torch.linalg.solve(K_a, k_atest.T)  # (n0, n0*N)
            F_aa = kpost_atest  # (n0*N,)
            G_aa = (k_atest* E_a).sum(0)  # (n0*N,)
    
            Theta1 = torch.linalg.solve(K_vv, R_vv0) @ torch.linalg.solve(R_v0v0, K_v0v0)  # (n1+n0, n1+n0)
            Theta4 = torch.linalg.solve(K_v1v1, R_v1v0) @ torch.linalg.solve(R_v1, Y)  # (n1+n0, 1)
            Theta2a = Theta4.T @ R_vv @ Theta4  # (1, 1)
            Theta2b = Theta4.T @ R_vv0 @ torch.linalg.solve(R_v0v0, R_vv0.T) @ Theta4  # (1, 1)
            Theta3a = torch.trace(torch.linalg.solve(K_vv, R_vv) @ torch.linalg.solve(K_vv, R_vv - R_vv1 @ torch.linalg.solve(R_v1, R_vv1.T)))  # scalar
            Theta3b = torch.trace(torch.linalg.solve(K_vv, R_vv0) @ torch.linalg.solve(R_v0v0, R_vv0.T) @ torch.linalg.solve(K_vv, R_vv - R_vv1 @ torch.linalg.solve(R_v1, R_vv1.T)))  # scalar
    
            V1 = ((Theta1.T @ (R_vv - R_vv1 @ torch.linalg.solve(R_v1, R_vv1.T)) @ Theta1 @ E_a)*E_a).sum(0)  # (n0*N, )
            V2 = Theta2a * F_aa - Theta2b * G_aa  # (n0*N,)
            V3 = Theta3a * F_aa - Theta3b * G_aa  # (n0*N,)
    
            posterior_variance = V1 + V2 + V3  # (n0*N,)
    
            if average_doA:
                posterior_variance = posterior_variance.reshape(n0, N).mean(0).reshape(N, 1)  # (N, 1)
            else:
                posterior_variance = posterior_variance.reshape(N, 1)  # (N, 1)
            return posterior_variance
    

        
            
            