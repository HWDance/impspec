import torch
from src.GP_utils import *
from src.kernel_utils import median_heuristic
from src.kernels import NuclearKernel

class causalKLGP:
    """
    causalKLGP method for estimating posterior moments of average causal effects
    """

    def __init__(self,Kernel_A, Kernel_V, Kernel_Z, dim_A, dim_V, samples):

        d,p = dim_V, dim_A
        
        # Initialising hypers        
        base_kernel_V = Kernel_V(lengthscale = torch.tensor([d**0.5*1.0]).repeat(d).requires_grad_(True), 
                                scale = torch.tensor([1.0], requires_grad = True))
        self.kernel_V = NuclearKernel(base_kernel_V, 
                                                 Normal(torch.zeros(d),torch.ones(d)),
                                                 samples)
        self.noise_Y = torch.tensor(-2.0, requires_grad = True).float()
        
        self.kernel_A = Kernel_A(lengthscale =torch.tensor([p**0.5*1.0]).repeat(p).requires_grad_(False),
                                  scale = torch.tensor([1.0], requires_grad = True))
        self.noise_feat = torch.tensor(-2.0, requires_grad = True)

    """Will eventually be specific to front and back door"""
    def train(self, Y, A, V, niter, learn_rate, reg = 1e-4, switch_grads_off = True, train_feature_lengthscale = True, force_PD = False):

        # Constructing list of V_1,V_2 for compatibility with data fusion case
        if type(V) != list:
            V = [V,V]
        n,d = V[1].size()
        Y = Y.reshape(n,)
        
        """Training P(Y|V)"""
        
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

        # Getting lengthscale
        self.kernel_A.lengthscale = median_heuristic(A)
        
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
            
    """Compute E[E[Y|do(A)]] in A -> V -> Y """
    def post_mean(self, Y, A, V, doA, reg = 1e-4):
        
        # Constructing list of V_1,V_2 for compatibility with data fusion case
        if type(V) != list:
            V = [V,V]
            
        n1,n0 = len(Y), len(A)
        Y = Y.reshape(n1,1)
        
        # getting kernel matrices
        K_v0v1,K_vv,K_aa,k_atest = (self.kernel_V.get_gram_base(V[0],V[1]),
                                    self.kernel_V.get_gram_base(V[1],V[1]),
                             self.kernel_A.get_gram(A,A),
                             self.kernel_A.get_gram(doA, A))
        K_v = K_vv+(torch.exp(self.noise_Y)+reg)*torch.eye(n1)
        K_a = K_aa+(torch.exp(self.noise_feat)+reg)*torch.eye(n0)

        # Getting components
        A_a = torch.linalg.solve(K_a,k_atest.T).T
        alpha_y = torch.linalg.solve(K_v,Y)
        
        return  A_a @ K_v0v1 @ alpha_y
        
    """Compute Var[E[Y|(do(A)]] in A -> V -> Y """
    def post_var(self, Y, A, V, doA, reg = 1e-4, latent = True, nu=1):
                
        # Constructing list of V_1,V_2 for compatibility with data fusion case
        if type(V) != list:
            V = [V,V]

        n1,n0 = len(Y),len(A)
        Y = Y.reshape(n1,1)

        # Updating nuclear dominant kernel
        self.kernel_V.dist.scale = nu*V[1].var(0)**0.5
        self.kernel_V.dist.loc = V[1].mean(0)

        
        # getting kernel matrices
        R_vv,K_v0v1,K_v0v0,K_v1v1,K_aa,k_atest = (self.kernel_V.get_gram_approx(V[1],V[1]),
                                 self.kernel_V.get_gram_base(V[0],V[1]),
                                 self.kernel_V.get_gram_base(V[0],V[0]),
                                 self.kernel_V.get_gram_base(V[1],V[1]),
                                 self.kernel_A.get_gram(A,A),
                                 self.kernel_A.get_gram(doA, A))
        K_v = K_v1v1+(torch.exp(self.noise_Y)+reg)*torch.eye(n1)
        K_a = K_aa+(torch.exp(self.noise_feat)+reg)*torch.eye(n0)
        kpost_atest = GPpostvar(A, doA, self.kernel_A, torch.exp(self.noise_feat), latent = latent)
        
        # computing matrix vector products
        alpha_a = torch.linalg.solve(K_a,k_atest.T)
        alpha_y = torch.linalg.solve(K_v,Y)
        KainvKvv = torch.linalg.solve(K_a,K_v0v1)
        B = torch.linalg.solve(K_v, R_vv)
        DDKvv =  torch.linalg.solve(K_v, K_v0v1.T)
        
        V1 = kpost_atest*(alpha_y.T @ R_vv @ alpha_y).view(1,)
        V2 = kpost_atest*(K_v0v0[0,0] - torch.trace(B))
        V3 = k_atest @ KainvKvv @ (torch.eye(n1) - DDKvv) @ alpha_a+torch.exp(self.noise_Y)*(not latent)*torch.eye(len(doA))
        
        return (V1+V2+V3).diag()[:,None]

    def nystrom_sample(self,Y,V,A,doA, reg = 1e-4, features = 100, samples = 10**3, nu = 1):

        ### NOT COMPATIBLE WITH TWO DATASETS
            
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
                              niter = 500, learn_rate = 0.1, reg = 1e-4, levels = [],
                              bootstrap_replications = 20, retrain_hypers = False, 
                              retrain_iters = 500, retrain_lr = 0.1, sample_split = False, train_cal_split = 0.5,
                              marginal_loss = False, seed=0): 
        
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
                       reg = reg, niter = niter, learn_rate = learn_rate, switch_grads_off = False)
            
            # Estimating posterior mean \theta(P_n)
            mean = self.post_mean(Ytr, Atr, Vtr, doA, reg = reg).detach()

            # Retraining hypers for calibration
            if retrain_hypers:
                self.train(Ycal, Acal, Vcal, 
                   reg = reg, niter = retrain_iters, learn_rate = retrain_lr, switch_grads_off = False)
            
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
                meanb = self.post_mean(Yb, Ab, Vb, doA, reg = reg).detach()

                # iterating over nulist, get post-var and indicator for is_inside_CI per level and do(Z)xdo(W)
                for k in range(len(nulist)):
                    varb = self.post_var(Yb, Ab, Vb, doA, reg = reg, latent = True, nu = nulist[k]).detach()
                    
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



    
