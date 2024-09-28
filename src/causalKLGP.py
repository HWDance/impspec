import torch
from src.GP_utils import *
from src.kernel_utils import median_heuristic
from src.kernels import NuclearKernel
from copy import deepcopy

class causalKLGP:
    """
    causalKLGP method for estimating posterior moments of average causal effects
    """

    def __init__(self,Kernel_A, Kernel_V, dim_A, dim_V, samples, 
                 lengthscale_V_init = 1.0, scale_V_init = 1.0, noise_Y_init = -1.0,
                 lengthscale_A_init = 1.0, scale_A_init = 1.0, noise_feat_init = -2.0):

        d,p = dim_V, dim_A
        
        # Initialising hypers        
        base_kernel_V = Kernel_V(lengthscale = torch.tensor([d**0.5*lengthscale_V_init]).repeat(d).requires_grad_(True), 
                                scale = torch.tensor([scale_V_init], requires_grad = True))
        self.kernel_V = NuclearKernel(base_kernel_V, 
                                                 Normal(torch.zeros(d),torch.ones(d)),
                                                 samples)
        self.noise_Y = torch.tensor(noise_Y_init, requires_grad = True).float()
        
        self.kernel_A = Kernel_A(lengthscale =torch.tensor([p**0.5*lengthscale_A_init]).repeat(p).requires_grad_(True),
                                  scale = torch.tensor([scale_A_init], requires_grad = True))
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
    def train(self, Y, A, V, niter, learn_rate, reg = 1e-4, switch_grads_off = True, train_feature_lengthscale = True, force_PD = False):

        # Constructing list of V_1,V_2 for compatibility with data fusion case
        if type(V) != list:
            V = [V,V]
        n,d = V[1].size()
        Y = Y.reshape(n,)
        
        """Training P(Y|V)"""

        self.kernel_V.base_kernel.lengthscale = median_heuristic(V[1], per_dimension = True)

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
            loss = -GPML(Y, V[1], self.kernel_V.base_kernel, torch.exp(self.noise_Y),
                        force_PD = force_PD)
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

        self.kernel_A.lengthscale = median_heuristic(A, per_dimension = True)
        
        # Optimiser set up
        params_list = [self.kernel_A.hypers,
                                      self.kernel_A.scale,
                                      self.noise_feat]
        if train_feature_lengthscale:
            self.kernel_A.lengthscale = self.kernel_A.lengthscale.requires_grad_(True)
            params_list.append(self.kernel_A.lengthscale)            
        optimizer = torch.optim.Adam(params_list, lr=learn_rate)
        Losses = torch.zeros(niter)
        
        # Updates
        for i in range(niter):
            optimizer.zero_grad()
            loss =  -GPmercerML(V[0], A, self.kernel_V, self.kernel_A, torch.exp(self.noise_feat),
                               force_PD = force_PD)
            Losses[i] = loss.detach()
            loss.backward()
            optimizer.step()
            if not i % 100:
                print("iter {0} P(V|A) loss: ".format(i), Losses[i])
    
        # Disabling gradients
        if switch_grads_off:
            for param in params_list:
                param = param.requires_grad_(False)
            
    """Compute mean of E[Y|do(A)] in A -> V -> Y """
    def post_mean(self, Y, A, V, doA, reg=1e-4, average_doA = False, intervention_indices=None):
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
        
        # Ensuring V is a list for compatibility with data fusion
        if type(V) != list:
            V = [V, V]
    
        n1, n0 = Y.shape[0], A.shape[0]
        N, D = doA.shape[0],A.shape[1]
        Y = Y.reshape(n1, 1)  # Ensure Y is a column vector

        # Getting intervention and average indices and constructing expanded doA
        if average_doA:
            average_indices = [j for j in range(D) if j not in intervention_indices]
            expanded_doA = self.expand_doA(doA, A, intervention_indices)  # Shape (n0*N, D)
        else:
            assert doA.shape[1] == A.shape[1]
            expanded_doA = doA  # No expansion if not averaging, expanded_doA: (N, D)
    
        # Getting kernel matrices
        K_v0v1 = self.kernel_V.get_gram_base(V[0], V[1])  # (n0, n1)
        K_vv = self.kernel_V.get_gram_base(V[1], V[1])    # (n1, n1)
        K_aa = self.kernel_A.get_gram(A, A)               # (n0, n0)
        k_atest = self.kernel_A.get_gram(expanded_doA, A)  # (n0*N, n0)
        K_v = K_vv + (torch.exp(self.noise_Y) + reg) * torch.eye(n1)  # (n1, n1)
        K_a = K_aa + (torch.exp(self.noise_feat) + reg) * torch.eye(n0)  # (n0, n0)

        # Averaging out selected indices
        if average_doA:
            k_atest = k_atest.reshape(n0,N,n0).mean(0)
    
        # Solving for the components
        beta_a = torch.linalg.solve(K_a, k_atest.T)  # (N, n0)
        alpha_y = torch.linalg.solve(K_v, Y)        # (n1, 1)
    
        return beta_a.T @ K_v0v1 @ alpha_y  # (N,1)
    
    """Compute var of E[Y|(do(A)] in A -> V -> Y """
    def post_var(self, Y, A, V, doA, doA2=[], reg=1e-4, latent=True, nu=1, average_doA=False, intervention_indices=None, diag = True):
        """
        Compute the posterior variance Var(E[Y|do(A=a)]) with selective averaging, 
        optimized kernel computation, and integration of the GPpostvar function.
        
        :param Y: Observations from D1, shape (n1,)
        :param A: Observations from D0, shape (n0, D)
        :param V: List of V observations [V0, V1] corresponding to D0 and D1, respectively
        :param doA: Intervention points, shape (N, D)
        :param reg: Regularization parameter
        :param latent: Boolean indicating whether to include latent variable contribution
        :param nu: Scale parameter for kernel
        :param average_doA: Whether to average over the intervention points
        :param average_indices: List of column indices of doA to average over using columns from A
        :return: Posterior variance, shape (n0, 1)
        """
    
        # Ensure V is a list for compatibility
        if type(V) != list:
            V = [V, V]
        
        # Getting second set of doA if computing covariance
        if doA2 == []:
            doA2 = doA 

        # Dimensions
        n1, n0 = Y.shape[0], A.shape[0]
        N, M, D = doA.shape[0], doA2.shape[0], A.shape[1]
        Y = Y.reshape(n1, 1)  # Reshape Y to (n1, 1) to ensure it's a column vector
    
        # Update the nuclear dominant kernel based on V1
        self.kernel_V.dist.scale = nu * V[1].var(0) ** 0.5  # Scale for the kernel on V1
        self.kernel_V.dist.loc = V[1].mean(0)  # Location for the kernel on V1
    
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

        # Compute kernel matrices
        R_vv = self.kernel_V.get_gram_approx(V[1], V[1])  # (n1, n1)
        K_v0v1 = self.kernel_V.get_gram_base(V[0], V[1])  # (n0, n1)
        K_v0v0 = self.kernel_V.get_gram_base(V[0], V[0])  # (n0, n0)
        K_v1v1 = self.kernel_V.get_gram_base(V[1], V[1])  # (n1, n1)
        K_aa = self.kernel_A.get_gram(A, A)               # (n0, n0)
        k_atest = self.kernel_A.get_gram(expanded_doA, A)  # (n0*N, n0) if expanded, or (N, n0)
        k_atest2 = self.kernel_A.get_gram(expanded_doA2, A)  # (n0*M, n0) if expanded, or (M, n0)
        K_v = K_v1v1 + (torch.exp(self.noise_Y) + reg) * torch.eye(n1)  # (n1, n1)
        K_a = K_aa + (torch.exp(self.noise_feat) + reg) * torch.eye(n0)  # (n0, n0)
        
        # Averaging out selected indices
        if average_doA:
            k_atest = k_atest.reshape(n0,N,n0).mean(0) # (N,n0)
            k_atest2 = k_atest2.reshape(n0,M,n0).mean(0) # (M,n0)
            
        # Solving the matrix vector products
        beta_a = torch.linalg.solve(K_a, k_atest.T)  # (n0, N)
        beta_a2 = torch.linalg.solve(K_a, k_atest2.T)  # (n0, M)
        alpha_y = torch.linalg.solve(K_v, Y)          # (n1, 1)
        KainvKvv = torch.linalg.solve(K_a, K_v0v1)    # (n0, n1)
        B = torch.linalg.solve(K_v, R_vv)             # (n1, n1)
        DDKvv = torch.linalg.solve(K_v, K_v0v1.T)     # (n1, n0)

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
            V1 = kpost_atest * (alpha_y.T @ R_vv @ alpha_y).view(-1)  # (N,M)
            V2 = kpost_atest * (K_v0v0[0, 0] - torch.trace(B))  # (N, M)
            V3 = beta_a.T @ K_v0v0 @ beta_a2 - beta_a.T @ K_v0v1 @  DDKvv @ beta_a2 # (N, M)
            if not latent:
                V3 += torch.exp(self.noise_Y)*(torch.cdist(doA,doA2)==0)  # (N, M)

            posterior_variance = V1 + V2 + V3
          
        else:
            kpost_atest = K_atestatest.diag() - (torch.linalg.solve(K_a, k_atest2.T) * k_atest.T).sum(0) # (N, )
            V1 = kpost_atest * (alpha_y.T @ R_vv @ alpha_y).view(-1)  # (N, )
            V2 = kpost_atest * (K_v0v0[0, 0] - torch.trace(B))  # (N, )
            V3 = ((K_v0v0 @ beta_a2 - K_v0v1 @  DDKvv @ beta_a2)*beta_a).sum(0) # (N, )
            if not latent:
                V3 += torch.exp(self.noise_Y)  # (N, )
        
            # Compute the final posterior variance
            posterior_variance = (V1 + V2 + V3).reshape(N,1)  # (N, 1)

        return posterior_variance # (N, 1) or (N, N)

    def nystrom_sample(self,Y,V,A,doA, reg = 1e-4, features = 100, samples = 10**3, nu = 1):

        ### NOT COMPATIBLE WITH TWO DATASETS OR AVERAGING
            
        # Set up
        n = len(Y)
        Y = Y.reshape(n,1)
        ntest = len(doA)
        
        # Updating nuclear dominant kernel
        self.kernel_V.dist.scale = nu*V.var(0)**0.5
        self.kernel_V.dist.loc = V.mean(0)
        
        # Getting gram matrices
        U = self.kernel_V.dist.sample((features,)).detach()
        K_uu = self.kernel_V.get_gram_base(U,U).detach()
        K_vu = self.kernel_V.get_gram_base(V,U).detach()
        K_vv = self.kernel_V.get_gram_base(V,V).detach()
        K_aa = self.kernel_A.get_gram(A,A).detach()
        
        # Getting eignefunctions and eigenvalues
        eigs,vecs = torch.linalg.eig(K_uu/samples)
        eigs,vecs = eigs.real.abs(), vecs.real
        
        # Extending to datapoints
        Phi_v = samples**-0.5*K_vu @ (vecs / eigs)
        Phi_v_tilde = Phi_v @ eigs.diag()**0.5
        Phi_v_tilde2 = Phi_v @ eigs.diag()
        
        # Getting moments of f
        K_v = K_vv + (torch.exp(self.noise_Y)+reg)*torch.eye(n)
        alpha_y = torch.linalg.solve(K_v,Y)
        alpha_v = torch.linalg.solve(K_v, Phi_v_tilde2)
        mu_f = (Phi_v_tilde2.T @ alpha_y).detach()  # nfeateres x 1
        C_f = (eigs.diag() - Phi_v_tilde2.T @ alpha_v).detach() # nfeatures x nfeatures
        C_f = 0.5*(C_f + C_f.T)
        eigs_f,vecs_f = torch.linalg.eig(C_f)
        eigs_f,vecs_f = eigs_f.real.abs(),vecs_f.real
        C_fhalf = vecs_f @ eigs_f.diag()**0.5
        
        # Getting moments of tilde phi_v
        k_atesta = self.kernel_A.get_gram(doA,A).detach()
        K_a = K_aa + (torch.exp(self.noise_feat)+reg)*torch.eye(n)
        kpost_atest = GPpostvar(A, doA, self.kernel_A, torch.exp(self.noise_feat), latent = True).detach()
        kpost_atest_noise = GPpostvar(A, doA, self.kernel_A, torch.exp(self.noise_feat), latent = False).detach()
        mu_l = (Phi_v.T @ torch.linalg.solve(K_a,k_atesta.T)).detach()  # nfeatures x ntest 
        C_l = kpost_atest.diag().detach() # ntest x 1 (dont need off-diagonals)
        C_l_noise = kpost_atest_noise.diag().detach() # ntest x 1 (dont need off-diagonals)
        
        # Getting samples of f
        #F_s = MultivariateNormal(mu_f,C_f).sample((samples,)) # nsamples x ntest
        epsilon = Normal(torch.zeros(features),torch.ones(features)).sample((samples,))
        F_s = mu_f.T+epsilon @ C_fhalf.T
        Phi_vs = Normal(mu_l,C_l).sample((samples,)) # nsamples x nfeat # ntest
        Phi_vs_noise = Normal(mu_l,C_l_noise).sample((samples,)) # nsamples x nfeat # ntest
        EYdoA_sample = (F_s[...,None]*Phi_vs).sum(1) # nsamples x ntest
        YdoA_sample = (F_s[...,None]*Phi_vs_noise).sum(1) + Normal(0,torch.exp(self.noise_Y)**0.5).sample((samples,ntest)) # nsamples x ntest

        return EYdoA_sample, YdoA_sample

    def frequentist_calibrate(self, Y, V, A, doA = [], nulist = [], 
                              niter = 500, learn_rate = 0.1, reg = 1e-4, force_PD = False, levels = [],
                              bootstrap_replications = 20, retrain_hypers = False, 
                              retrain_iters = 500, retrain_lr = 0.1, sample_split = False, train_cal_split = 0.5,
                              marginal_loss = False, seed=0, average_doA = False, intervention_indices = None): 
        
            # Set up
            if type(V) == list:
                two_datasets = True
            else:
                V = [V,V]
                two_datasets = False

            if levels == []:
                levels = torch.linspace(0,1,101)        
            levels = levels.reshape(len(levels),)
            z_quantiles = Normal(0, 1).icdf(1-(1-levels)/2).reshape(len(levels),1)
            n1,n0 = len(Y),len(A)

            # Specifying base datasets (either by sample splitting or not)
            if sample_split:
                torch.manual_seed(seed)
                shuffle1 = torch.randperm(n1)
                if two_datasets:
                    shuffle0 = torch.randperm(n0)
                else:
                    shuffle0 = shuffle1
                ntr0 = int(n0*train_cal_split)
                ntr1 = int(n1*train_cal_split)
                ncal0 = n0-ntr0
                ncal1 = n1-ntr1
                Ytr,Vtr,Atr = Y[shuffle1][:ntr1],[V[0][shuffle0][:ntr0],V[1][shuffle1][:ntr1]],A[shuffle0][:ntr0]
                Ycal,Vcal,Acal = Y[shuffle1][ntr1:],[V[0][shuffle0][ntr0:],V[1][shuffle1][ntr1:]],A[shuffle0][ntr0:]
            else:
                Ytr,Vtr,Atr  = Y,V,A
                Ycal,Vcal,Acal = Y,V,A
                ncal1,ncal0 = n1,n0
        
            # Training model (for theta(Pn))
            self.train(Ytr, Atr, Vtr, 
                       reg = reg, niter = niter, learn_rate = learn_rate, switch_grads_off = False,
                      force_PD = force_PD)
            
            # Estimating posterior mean \theta(P_n)
            mean = self.post_mean(Ytr, Atr, Vtr, doA, reg = reg, 
                                  average_doA = average_doA, intervention_indices = intervention_indices).detach()

            # Retraining hypers for calibration
            if retrain_hypers:
                self.train(Ycal, Acal, Vcal, 
                   reg = reg, niter = retrain_iters, learn_rate = retrain_lr, switch_grads_off = False,
                          force_PD = force_PD)
            
            # Getting bootstrap indices
            bootstrap_inds1 = [torch.randint(0, ncal1, (ncal1,)) for _ in range(bootstrap_replications)]
            if two_datasets:
                bootstrap_inds0 = [torch.randint(0, ncal0, (ncal0,)) for _ in range(bootstrap_replications)]
            else:
                bootstrap_inds0 = bootstrap_inds1
                

            # Looping over calibration parameter values and bootstrap datasets
            Post_levels = torch.zeros((len(nulist),len(doA), len(levels)))
            for b in range(len(bootstrap_inds1)):
                    
                # Getting dataset
                Yb, Vb, Ab, = Ycal[bootstrap_inds1[b]], [Vcal[0][bootstrap_inds0[b]],Vcal[1][bootstrap_inds1[b]]], A[bootstrap_inds0[b]]
                                    
                # Get posterior mean
                meanb = self.post_mean(Yb, Ab, Vb, doA, reg = reg,
                                       average_doA = average_doA, intervention_indices = intervention_indices).detach()

                # iterating over nulist, get post-var and indicator for is_inside_CI per level and do(Z)xdo(W)
                for k in range(len(nulist)):
                    varb = self.post_var(Yb, Ab, Vb, doA, reg = reg, latent = True, nu = nulist[k], 
                                         average_doA = average_doA, intervention_indices = intervention_indices).detach()
                    
                    # Get posterior coverage
                    is_inside_region = ((mean - meanb).abs() <= varb**0.5 @ z_quantiles.T).float() # len(doZ) x len(levels)
                    Post_levels[k] += is_inside_region/bootstrap_replications

            # Scoring best model
            Calibration_losses = torch.zeros(len(nulist))
        
            for k in range(len(nulist)):
                if marginal_loss:
                    Calibration_losses[k] = (Post_levels[k].mean(0) - levels).abs().mean()
                else:
                    Calibration_losses[k] = (Post_levels[k] - levels).abs().mean()                  

            return Post_levels, Calibration_losses



    
