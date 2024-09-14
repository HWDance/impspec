import torch
from src.GP_utils import *
from src.kernel_utils import median_heuristic
from src.kernels import NuclearKernel, ProductKernel

class causalKLGP:
    """
    causalKLGP method for estimating posterior moments of average causal effects
    """

    def __init__(self, Kernel_V, Kernel_W, Kernel_Z, dim_V, dim_W, dim_Z, samples):

        d,k,p =  dim_V, dim_W, dim_Z
        
        # Initialising kernels and hypers        
        base_kernel_V = Kernel_V(lengthscale = torch.tensor([d**0.5*1.0]).repeat(d).requires_grad_(True), 
                                scale = torch.tensor([1.0], requires_grad = True))
        self.kernel_V = NuclearKernel(base_kernel_V, 
                                                 Normal(torch.zeros(d),torch.ones(d)),
                                                 samples)
        if Kernel_W != []:
            self.kernel_W =  Kernel_W(lengthscale = torch.tensor([k**0.5*1.0]).repeat(k).requires_grad_(True), 
                                scale = torch.tensor([1.0], requires_grad = False))
            self.kernel_WV = ProductKernel(self.kernel_W, self.kernel_V.base_kernel)
        
        else:
            self.kernel_W = []
            self.kernel_WV = self.kernel_V.base_kernel
        self.noise_Y = torch.tensor(-2.0, requires_grad = True).float()
        
        if Kernel_Z != []:
            self.kernel_Z = Kernel_Z(lengthscale =torch.tensor([p**0.5*1.0]).repeat(p).requires_grad_(False),
                                  scale = torch.tensor([1.0], requires_grad = True))
        else:
            self.kernel_Z = []
        self.noise_feat = torch.tensor(-2.0, requires_grad = True)
        

    def train(self, Y, V, W=[], Z=[], niter = 500, learn_rate = 0.1, reg = 1e-4, switch_grads_off = True, train_feature_lengthscale = False):
    
        """Training P(Y|V,W)"""
        n,d = V.size()
        Y = Y.reshape(n,)
        
        # Optimiser set up
        params_list = [self.kernel_V.base_kernel.lengthscale,
                                      self.kernel_V.base_kernel.scale,
                                      self.kernel_V.base_kernel.hypers,
                                      self.noise_Y]
        # If including W in model (V,W) -> Y, construct product kernel to optimise
        if W !=[]:
            WV = [W,V]
            params_list.extend([self.kernel_W.lengthscale,
                                      self.kernel_W.scale,
                                      self.kernel_W.hypers])
        else:
            WV = V
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
        if switch_grads_off:
            for param in params_list:
                param = param.requires_grad_(False)
            
        """
        Training P(V|Z)
        """
        if Z!=[]:
            n,p = Z.size()
    
            # Getting lengthscale
            self.kernel_Z.lengthscale = median_heuristic(Z)
            
            # Optimiser set up
            params_list = [self.kernel_Z.hypers,
                                          self.kernel_Z.scale,
                                          self.noise_feat]
            if train_feature_lengthscale:
                self.kernel_Z.lengthscale = self.kernel_Z.lengthscale.requires_grad_(True)
                params_list.append(self.kernel_Z.lengthscale)            
            optimizer = torch.optim.Adam(params_list, lr=learn_rate)
            Losses = torch.zeros(niter)
            
            # Updates
            for i in range(niter):
                optimizer.zero_grad()
                loss =  -GPmercerML(V, Z, self.kernel_V, self.kernel_Z, torch.exp(self.noise_feat))
                Losses[i] = loss.detach()
                loss.backward()
                optimizer.step()
                if not i % 100:
                    print("iter {0} P(V|Z) loss: ".format(i), Losses[i])
        
            # Disabling gradients
            if switch_grads_off:
                for param in params_list:
                    param = param.requires_grad_(False)
            
    """Compute E[E[Y|do(W), do(Z)]] in Z -> V, (W,V) -> Y """
    def post_mean(self, Y, V, W = [], Z = [], doW = [], doZ = [], reg = 1e-4):
        
        assert(doW != [] or doZ != [])

        
        n = len(Y)
        Y = Y.reshape(n,1)
        if W != []:
            WV = [W,V]
        else:
            WV = V
        # getting kernel matrices
        K_wvwv, K_vv,K_zz,k_ztest = (self.kernel_WV.get_gram(WV,WV),
                                      self.kernel_V.get_gram_base(V,V),
                                     self.kernel_Z.get_gram(Z,Z),
                                     self.kernel_Z.get_gram(Z, doZ))
        K_wv = K_wvwv+(torch.exp(self.noise_Y)+reg)*torch.eye(n)
        K_z = K_zz+(torch.exp(self.noise_feat)+reg)*torch.eye(n)

        # Getting components
        alpha_y = torch.linalg.solve(K_wv,Y) # n x 1
        if W != []:
            alpha_y = alpha_y*self.kernel_W.get_gram(W,doW) # n x ntest_w
        beta_z = torch.linalg.solve(K_z,k_ztest) # n x ntest
        
        return  (beta_z.T @ K_vv @ alpha_y) # ntest_z x ntest_w
        
    """Compute Var[E[Y|(do(Z)]] in Z -> V -> Y """
    def post_var(self, Y, V, W = [], Z = [], doW = [], doZ = [], reg = 1e-4, latent = True, nu=1):
    
        n = len(Y)
        n_w, n_z = max(len(doW),1), len(doZ)
        Y = Y.reshape(n,1)

        # Updating nuclear dominant kernel
        self.kernel_V.dist.scale = nu*V.var(0)**0.5
        self.kernel_V.dist.loc = V.mean(0)

        # getting kernel matrices
        if W != []:
            WV = [W,V]
        else:
            WV = V
        R_vv,K_wvwv,K_vv,K_zz,k_ztest = (self.kernel_V.get_gram_approx(V,V),
                                 self.kernel_WV.get_gram(WV,WV),
                                self.kernel_V.get_gram_base(V,V),
                                 self.kernel_Z.get_gram(Z,Z),
                                 self.kernel_Z.get_gram(Z, doZ))
        K_wv = K_wvwv+(torch.exp(self.noise_Y)+reg)*torch.eye(n)
        K_z = K_zz+(torch.exp(self.noise_feat)+reg)*torch.eye(n)
        kpost_ztest = GPpostvar(Z, doZ, self.kernel_Z, torch.exp(self.noise_feat), latent = latent).diag() # ntest x 0
        if W != []:
            k_wtest = self.kernel_W.get_gram(W,doW)
            kwtestwtest = self.kernel_W.get_gram(doW,doW).diag()
        else:
            k_wtest = torch.ones((n,1))
            kwtestwtest = torch.ones((1,1))
            
        # Getting components
        alpha_y = torch.linalg.solve(K_wv,Y) # n x 1
        beta_z = torch.linalg.solve(K_z,k_ztest) # n x n_z
        bzKvv = beta_z.T @ K_vv # n_z x n
        bzKvvbz = (bzKvv @ beta_z).diag() # n_z x 0
            

        # Getting variance terms by looping over w
        V1,V2a,V2b,V3a,V3b = (torch.zeros((n_z, n_w)),
                               torch.zeros((n_z, n_w)),
                               torch.zeros((n_z, n_w)),
                               torch.zeros((n_z, n_w)),
                               torch.zeros((n_z, n_w)))        
        for w in range(n_w):
            
            # Getting V1
            alpha_a = alpha_y*k_wtest[:,w].reshape(n,1)
            alphayRalphay = (alpha_a.T @ R_vv @ alpha_a).diag() # 1 x 0
            V1[:,w] = kpost_ztest*alphayRalphay # n_z x  0
    
            # Getting V2(a)
            V2a[:,w] = bzKvvbz*kwtestwtest[w,w] # # n_z x 0
    
            # Getting V2(b)
            bzKvvDw = bzKvv * k_wtest[:,w][None] # n_z x n
            V2b[:,w] = -(bzKvvDw @ torch.linalg.solve(K_wv, bzKvvDw.T)).diag() # ntest x 0
    
            # Getting V3(a)
            V3a[:,w] = kpost_ztest*K_vv[0,0]*kwtestwtest[w,w] # ntest x 0

            # Getting V3b
            B = torch.linalg.solve(K_wv,  k_wtest[:,w].diag() @ R_vv @  k_wtest[:,w].diag())
            V3b[:,w] = -kpost_ztest*torch.trace(B)
            
        return V1+V2a+V2b+V3a+V3b+torch.exp(self.noise_Y)*(not latent)

    def nystrom_sample(self,Y, V, W=[], Z = [], doW = [], doZ = [], reg = 1e-4, features = 100, samples = 10**3, nu = 1):

        # Set up
        n = len(Y)
        Y = Y.reshape(n,1)
        n_z,n_w = len(doZ), max(len(doW),1)
        if W != []:
            WV = [W,V]
        else:
            WV = V
            
        # Updating nuclear dominant kernel
        self.kernel_V.dist.scale = nu*V.var(0)**0.5
        self.kernel_V.dist.loc = V.mean(0)
        
        # Getting gram matrices
        U = self.kernel_V.dist.sample((features,)).detach()
        K_uu = self.kernel_V.get_gram_base(U,U).detach()
        K_vu = self.kernel_V.get_gram_base(V,U).detach()
        K_wvwv = self.kernel_WV.get_gram(WV,WV).detach()
        K_zz = self.kernel_Z.get_gram(Z,Z).detach()
        if W != []:
            k_wtest = self.kernel_W.get_gram(W,doW)
            kwtestwtest = self.kernel_W.get_gram(doW,doW).diag()
        else:
            k_wtest = torch.ones((n,1))
            kwtestwtest = torch.ones((1,1))

        # Getting eignefunctions and eigenvalues
        eigs,vecs = torch.linalg.eig(K_uu/samples)
        eigs,vecs = eigs.real.abs(), vecs.real
        
        # Extending to datapoints
        Phi_v = samples**-0.5*K_vu @ (vecs / eigs)
        Phi_v_tilde = Phi_v @ eigs.diag()**0.5
        Phi_v_tilde2 = Phi_v @ eigs.diag()

        # Getting moments of tilde phi_v
        k_ztesta = self.kernel_Z.get_gram(doZ,Z).detach()
        K_z = K_zz + (torch.exp(self.noise_feat)+reg)*torch.eye(n)
        kpost_ztest = GPpostvar(Z, doZ, self.kernel_Z, torch.exp(self.noise_feat), latent = True).detach()
        kpost_ztest_noise = GPpostvar(Z, doZ, self.kernel_Z, torch.exp(self.noise_feat), latent = False).detach()
        mu_l = (Phi_v.T @ torch.linalg.solve(K_z,k_ztesta.T)).detach()  # nfeatures x ntest 
        C_l = kpost_ztest.diag().detach() # ntest x 1 (dont need off-diagonals)
        C_l_noise = kpost_ztest_noise.diag().detach() # ntest x 1 (dont need off-diagonals)
        
        # Getting moments of f
        K_wv = K_wvwv + (torch.exp(self.noise_Y)+reg)*torch.eye(n)
        alpha_y = torch.linalg.solve(K_wv,Y)

        # (still getting f moments, but from here loop over doW values)
        EYdoZ_sample,YdoZ_sample = (torch.zeros((samples, n_z, n_w)),
                                    torch.zeros((samples, n_z, n_w)))
        for w in range(n_w):
            alpha_v = torch.linalg.solve(K_wv, k_wtest[:,w].diag() @ Phi_v_tilde2)
            mu_f = (Phi_v_tilde2.T @ k_wtest[:,w].diag() @ alpha_y).detach()  # nfeateres x 1
            C_f = (eigs.diag()*kwtestwtest[w,w] - Phi_v_tilde2.T @ k_wtest[:,w].diag() @ alpha_v).detach() # nfeatures x nfeatures
            C_f = 0.5*(C_f + C_f.T)
            eigs_f,vecs_f = torch.linalg.eig(C_f)
            eigs_f,vecs_f = eigs_f.real.abs(),vecs_f.real
            C_fhalf = vecs_f @ eigs_f.diag()**0.5
            
            # Getting samples of f
            #F_s = MultivariateNormal(mu_f,C_f).sample((samples,)) # nsamples x ntest
            epsilon = Normal(torch.zeros(features),torch.ones(features)).sample((samples,))
            F_s = mu_f.T+epsilon @ C_fhalf.T
            Phi_vs = Normal(mu_l,C_l).sample((samples,)) # nsamples x nfeat # ntest
            Phi_vs_noise = Normal(mu_l,C_l_noise).sample((samples,)) # nsamples x nfeat # ntest
            EYdoZ_sample[...,w] = (F_s[...,None]*Phi_vs).sum(1) # nsamples x ntest
            YdoZ_sample[...,w] = (F_s[...,None]*Phi_vs_noise).sum(1) + Normal(0,torch.exp(self.noise_Y)**0.5).sample((samples,n_z)) # nsamples x ntest

        return EYdoZ_sample, YdoZ_sample
        
    def calibrate(self,Y, V, W, Z, nulist, niter, learn_rate, reg = 1e-4,  train_feature_lengthscale = False, train_cal_split=0.5, levels = [], seed=0, 
                  nystrom = False, nystrom_features = 100, nystrom_samples = 10**3, calibrate_latent = False, calibrate_norm = 1, train_calibration_model = False):
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
        Ytr,Vtr,Ztr = Y[shuffle][:ntr],V[shuffle][:ntr],Z[shuffle][:ntr]
        Ycal,Vcal,Zcal = Y[shuffle][ntr:],V[shuffle][ntr:],Z[shuffle][ntr:]
        if W!= []:
            Wtr,Wcal =  W[shuffle][:ntr], W[shuffle][ntr:]
        else:
            Wtr,Wcal = [],[]

        # Optionally train calibration model and get cal_f
        if calibrate_latent:
            if train_calibration_model:
                self.train(Ycal, Vcal, Wcal, Zcal, reg = reg, niter = niter, learn_rate = learn_rate, switch_grads_off = False, 
                           train_feature_lengthscale = train_feature_lengthscale)
            Ycal = self.post_mean(Ycal, Vcal, Wcal, Zcal, Wcal, Zcal, reg = reg).detach()


        # Training model and getting post mean
        self.train(Ytr, Vtr, Wtr, Ztr, reg = reg, niter = niter, learn_rate = learn_rate, switch_grads_off = False,
                  train_feature_lengthscale = train_feature_lengthscale)
        mean = self.post_mean(Ytr, Vtr, Wtr, Ztr, Wcal, Zcal, reg = reg).detach()

        
        # Iterating over hyperlist
        Calibration_losses= torch.zeros(len(nulist))
        Post_levels = []

    
        for k in range(len(nulist)):
            if not nystrom:

                if calibrate_latent:
                    var = self.post_var(Ytr, Vtr, Wtr, Ztr, Wcal, Zcal, reg = reg, latent = True, nu = nulist[k]).detach().diag()[:,None]
                    post_levels = GP_cal(Ycal, mean, var, levels[:,None])
                else:
                    var_noise = self.post_var(Ytr, Vtr, Wtr, Ztr, Wcal, Zcal, reg = reg, latent = False, nu = nulist[k]).detach().diag()[:,None]
                    post_levels = GP_cal(Ycal, mean, var_noise, levels[:,None])
            else:
                EYdoA_sample, YdoA_sample = self.nystrom_sample(Ytr,Vtr,Wtr, Ztr, Wcal, Zcal, reg, nystrom_features, nystrom_samples, nulist[k])
                upper_quantiles = 1-(1-levels)/2
                lower_quantiles = (1-levels)/2
                u = (upper_quantiles*(nystrom_samples-1)).int()
                l = (lower_quantiles*(nystrom_samples-1)).int()
                if calibrate_latent:
                    Y_u = EYdoA_sample[...,0].sort(0)[0][u]
                    Y_l = EYdoA_sample[...,0].sort(0)[0][l]
                else:
                    Y_u = YdoA_sample[...,0].sort(0)[0][u]
                    Y_l = YdoA_sample[...,0].sort(0)[0][l]
                post_levels = ((Y_u>=Ycal[:,0])*(Y_l<=Ycal[:,0])).float().mean(1)
                
            Post_levels.append(post_levels)
            Calibration_losses[k] = ((post_levels-levels[:,None].T).abs()**calibrate_norm).mean()     
  
        
        return Post_levels, Calibration_losses

 