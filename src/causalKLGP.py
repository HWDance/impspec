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
    def train(self, Y, A, V, niter, learn_rate, reg = 1e-4, switch_grads_off = True, train_feature_lengthscale = True):
    
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
            loss = -GPML(Y, V, self.kernel_V.base_kernel, torch.exp(self.noise_Y))
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
            loss =  -GPmercerML(V, A, self.kernel_V, self.kernel_A, torch.exp(self.noise_feat))
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
        
        n = len(Y)
        Y = Y.reshape(n,1)
        
        # getting kernel matrices
        K_vv,K_aa,k_atest = (self.kernel_V.get_gram_base(V,V),
                             self.kernel_A.get_gram(A,A),
                             self.kernel_A.get_gram(doA, A))
        K_v = K_vv+(torch.exp(self.noise_Y)+reg)*torch.eye(n)
        K_a = K_aa+(torch.exp(self.noise_feat)+reg)*torch.eye(n)

        # Getting components
        A_a = torch.linalg.solve(K_a,k_atest.T).T
        alpha_y = torch.linalg.solve(K_v,Y)
        
        return  A_a @ K_vv @ alpha_y
        
    """Compute Var[E[Y|(do(A)]] in A -> V -> Y """
    def post_var(self, Y, A, V, doA, reg = 1e-4, latent = True, nu=1):

        n = len(Y)
        Y = Y.reshape(n,1)

        # Updating nuclear dominant kernel
        self.kernel_V.dist.scale = nu*V.var(0)**0.5
        self.kernel_V.dist.loc = V.mean(0)

        
        # getting kernel matrices
        R_vv,K_vv,K_aa,k_atest = (self.kernel_V.get_gram_approx(V,V),
                                 self.kernel_V.get_gram_base(V,V),
                                 self.kernel_A.get_gram(A,A),
                                 self.kernel_A.get_gram(doA, A))
        K_v = K_vv+(torch.exp(self.noise_Y)+reg)*torch.eye(n)
        K_a = K_aa+(torch.exp(self.noise_feat)+reg)*torch.eye(n)
        kpost_atest = GPpostvar(A, doA, self.kernel_A, torch.exp(self.noise_feat), latent = latent)
        
        # computing matrix vector products
        alpha_a = torch.linalg.solve(K_a,k_atest.T)
        alpha_y = torch.linalg.solve(K_v,Y)
        KainvKvv = torch.linalg.solve(K_a,K_vv)
        B = torch.linalg.solve(K_v, R_vv)
        DDKvv =  torch.linalg.solve(K_v, K_vv)
        
        V1 = kpost_atest*(alpha_y.T @ R_vv @ alpha_y).view(1,)
        V2 = kpost_atest*(K_vv[0,0] - torch.trace(B))
        V3 = k_atest @ KainvKvv @ (torch.eye(n) - DDKvv) @ alpha_a+torch.exp(self.noise_Y)*(not latent)*torch.eye(len(doA))
        
        return V1+V2+V3

    def nystrom_sample(self,Y,V,A,doA, reg = 1e-4, features = 100, samples = 10**3, nu = 1):

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


    def calibrate(self,Y, V, A, nulist, niter, learn_rate, reg = 1e-4,  train_feature_lengthscale = False, train_cal_split=0.5, levels = [], seed=0, nystrom = False, nystrom_features = 100, nystrom_samples = 10**3, calibrate_latent = False, calibrate_norm = 1, train_calibration_model = False):
        """
        train_args = (niter,learn_rate,reg)

        Returns: nu hyperparameter for computing posterior
        """

        # Getting data splits
        n = len(Y)
        Y = Y.reshape(n,1)
        torch.manual_seed(seed)
        shuffle = torch.randperm(n)
        ntr = int(n*train_cal_split)
        Ytr,Vtr,Atr = Y[shuffle][:ntr],V[shuffle][:ntr],A[shuffle][:ntr]
        Ycal,Vcal,Acal = Y[shuffle][ntr:],V[shuffle][ntr:],A[shuffle][ntr:]

        # Optionally train calibration model and get cal_f
        if calibrate_latent:
            if train_calibration_model:
                self.train(Ycal, Acal, Vcal, reg = reg, niter = niter, learn_rate = learn_rate, switch_grads_off = False, 
                           train_feature_lengthscale = train_feature_lengthscale)
            Ycal = self.post_mean(Ycal, Acal, Vcal, Acal, reg = reg).detach()


        # Training model and getting post mean
        self.train(Ytr, Atr, Vtr, reg = reg, niter = niter, learn_rate = learn_rate, switch_grads_off = False,
                  train_feature_lengthscale = train_feature_lengthscale)
        mean = self.post_mean(Ytr, Atr, Vtr, Acal, reg = reg).detach()

        
        # Iterating over hyperlist
        Calibration_losses= torch.zeros(len(nulist))
        Post_levels = []

    
        for k in range(len(nulist)):
            if not nystrom:

                if calibrate_latent:
                    var = self.post_var(Ytr, Atr, Vtr, Acal, reg = reg, latent = True, nu = nulist[k]).detach().diag()[:,None]
                    post_levels = GP_cal(Ycal, mean, var, levels[:,None])
                else:
                    var_noise = self.post_var(Ytr, Atr, Vtr, Acal, reg = reg, latent = False, nu = nulist[k]).detach().diag()[:,None]
                    post_levels = GP_cal(Ycal, mean, var_noise, levels[:,None])
            else:
                EYdoA_sample, YdoA_sample = self.nystrom_sample(Ytr,Vtr,Atr,Acal,reg, nystrom_features, nystrom_samples, nulist[k])
                upper_quantiles = 1-(1-levels)/2
                lower_quantiles = (1-levels)/2
                u = (upper_quantiles*(nystrom_samples-1)).int()
                l = (lower_quantiles*(nystrom_samples-1)).int()
                if calibrate_latent:
                    Y_u = EYdoA_sample.sort(0)[0][u]
                    Y_l = EYdoA_sample.sort(0)[0][l]
                else:
                    Y_u = YdoA_sample.sort(0)[0][u]
                    Y_l = YdoA_sample.sort(0)[0][l]
                post_levels = ((Y_u>=Ycal[:,0])*(Y_l<=Ycal[:,0])).float().mean(1)
                
            Post_levels.append(post_levels)
            Calibration_losses[k] = ((post_levels-levels[:,None].T).abs()**calibrate_norm).mean()     
  
        
        return Post_levels, Calibration_losses




    
