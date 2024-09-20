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
        

    def train(self, Y, V, W=[], Z=[], niter = 500, learn_rate = 0.1, reg = 1e-4, switch_grads_off = True, train_feature_lengthscale = True, force_PD = False):

        # Compatibility with data fusion case
        if type(V) != list:
            V = [V,V]
        n,d = V[1].size()
        Y = Y.reshape(n,)
        
        
        """Training P(Y|V,W)"""        
        # Optimiser set up
        params_list = [self.kernel_V.base_kernel.lengthscale,
                                      self.kernel_V.base_kernel.scale,
                                      self.kernel_V.base_kernel.hypers,
                                      self.noise_Y]
        # If including W in model (V,W) -> Y, construct product kernel to optimise
        if W !=[]:
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
            loss = -GPML(Y, WV, self.kernel_WV, torch.exp(self.noise_Y), 
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
                loss =  -GPmercerML(V[0], Z, self.kernel_V, self.kernel_Z, torch.exp(self.noise_feat),
                                   force_PD = force_PD)
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
        
        # Constructing list of V_1,V_2 for compatibility with data fusion case
        if type(V) != list:
            V = [V,V]      
        
        n1,n0 = len(Y),len(Z)
        Y = Y.reshape(n1,1)
        if W != []:
            WV = [W,V[1]]
        else:
            WV = V[1]
        # getting kernel matrices
        K_wvwv, K_vv,K_zz,k_ztest = (self.kernel_WV.get_gram(WV,WV),
                                      self.kernel_V.get_gram_base(V[0],V[1]),
                                     self.kernel_Z.get_gram(Z,Z),
                                     self.kernel_Z.get_gram(Z, doZ))
        K_wv = K_wvwv+(torch.exp(self.noise_Y)+reg)*torch.eye(n1)
        K_z = K_zz+(torch.exp(self.noise_feat)+reg)*torch.eye(n0)

        # Getting components
        alpha_y = torch.linalg.solve(K_wv,Y) # n x 1
        if W != []:
            alpha_y = alpha_y*self.kernel_W.get_gram(W,doW) # n x ntest_w
        beta_z = torch.linalg.solve(K_z,k_ztest) # n x ntest

        return  (beta_z.T @ K_vv @ alpha_y) # ntest_z x ntest_w
        
    """Compute Var[E[Y|(do(Z)]] in Z -> V -> Y """
    def post_var(self, Y, V, W = [], Z = [], doW = [], doZ = [], reg = 1e-4, latent = True, nu=1):
             
        # Constructing list of V_1,V_2 for compatibility with data fusion case
        if type(V) != list:
            V = [V,V]
            
        n1,n0 = len(Y), len(Z)
        n_w, n_z = max(len(doW),1), len(doZ)
        Y = Y.reshape(n1,1)

        # Updating nuclear dominant kernel
        self.kernel_V.dist.scale = nu*V[1].var(0)**0.5
        self.kernel_V.dist.loc = V[1].mean(0)

        # getting kernel matrices
        if W != []:
            WV = [W,V[1]]
        else:
            WV = V[1]
        R_vv,K_wvwv,K_v0v1,K_v0v0,K_zz,k_ztest = (self.kernel_V.get_gram_approx(V[1],V[1]),
                                 self.kernel_WV.get_gram(WV,WV),
                                self.kernel_V.get_gram_base(V[0],V[1]),
                                self.kernel_V.get_gram_base(V[0],V[0]),
                                self.kernel_Z.get_gram(Z,Z),
                                 self.kernel_Z.get_gram(Z, doZ))
        K_wv = K_wvwv+(torch.exp(self.noise_Y)+reg)*torch.eye(n1)
        K_z = K_zz+(torch.exp(self.noise_feat)+reg)*torch.eye(n0)
        kpost_ztest = GPpostvar(Z, doZ, self.kernel_Z, torch.exp(self.noise_feat), latent = latent).diag() # ntest x 0
        if W != []:
            k_wtest = self.kernel_W.get_gram(W,doW)
            kwtestwtest = self.kernel_W.get_gram(doW,doW).diag()
        else:
            k_wtest = torch.ones((n1,1))
            kwtestwtest = torch.ones((1,1))
            
        # Getting components
        alpha_y = torch.linalg.solve(K_wv,Y) # n x 1
        beta_z = torch.linalg.solve(K_z,k_ztest) # n x n_z
        bzKvv = beta_z.T @ K_v0v1 # n_z x n
        bzKvvbz = (beta_z @ K_v0v0 @ beta_z).diag() # n_z x 0
            

        # Getting variance terms by looping over w
        V1,V2a,V2b,V3a,V3b = (torch.zeros((n_z, n_w)),
                               torch.zeros((n_z, n_w)),
                               torch.zeros((n_z, n_w)),
                               torch.zeros((n_z, n_w)),
                               torch.zeros((n_z, n_w)))        
        for w in range(n_w):
            
            # Getting V1
            alpha_a = alpha_y*k_wtest[:,w].reshape(n1,1)
            alphayRalphay = (alpha_a.T @ R_vv @ alpha_a).diag() # 1 x 0
            V1[:,w] = kpost_ztest*alphayRalphay # n_z x  0
    
            # Getting V2(a)
            V2a[:,w] = bzKvvbz*kwtestwtest[w,w] # # n_z x 0
    
            # Getting V2(b)
            bzKvvDw = bzKvv * k_wtest[:,w][None] # n_z x n
            V2b[:,w] = -(bzKvvDw @ torch.linalg.solve(K_wv, bzKvvDw.T)).diag() # ntest x 0
    
            # Getting V3(a)
            V3a[:,w] = kpost_ztest*K_v0v0[0,0]*kwtestwtest[w,w] # ntest x 0

            # Getting V3b
            B = torch.linalg.solve(K_wv,  k_wtest[:,w].diag() @ R_vv @  k_wtest[:,w].diag())
            V3b[:,w] = -kpost_ztest*torch.trace(B)
            
        return V1+V2a+V2b+V3a+V3b+torch.exp(self.noise_Y)*(not latent)

    def nystrom_sample(self,Y, V, W=[], Z = [], doW = [], doZ = [], reg = 1e-4, features = 100, samples = 10**3, nu = 1):

        ### NOT COMPATIBLE WITH TWO DATASETS

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
        
    def frequentist_calibrate(self, Y , V, W=[], Z=[], doW = [], doZ = [], 
                              nulist = [], niter = 500, learn_rate = 0.1, reg = 1e-4, 
                              levels = [], seed=0, bootstrap_replications = 20, retrain_hypers = False,
                              retrain_iters = 500, retrain_lr = 0.1, sample_split = False, train_cal_split = 0.5,
                              marginal_loss = False, seed = 0):
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
            n1,n0 = len(Y),len(Z)

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
                Ytr,Vtr,Ztr = Y[shuffle1][:ntr1],[V[0][shuffle0][:ntr0],V[1][shuffle1][:ntr1]],Z[shuffle0][:ntr0]
                Ycal,Vcal,Zcal = Y[shuffle1][ntr1:],[V[0][shuffle0][ntr0:],V[1][shuffle1][ntr1:]],Z[shuffle0][ntr0:]
                if W!= []:
                    Wtr,Wcal =  W[shuffle1][:ntr1], W[shuffle1][ntr1:]
                else:
                    Wtr,Wcal = [],[]
            else:
                Ytr,Vtr,Wtr,Ztr  = Y,V,W,Z
                Ycal,Vcal,Wcal,Zcal = Y,V,W,Z
                ncal1,ncal0 = n1,n0
                
        
            # Training model for theta(Pn)
            self.train(Ytr, Vtr, Wtr, Ztr, 
                       reg = reg, niter = niter, learn_rate = learn_rate, switch_grads_off = False)
            
            # Estimating posterior mean \theta(P_n)
            mean = self.post_mean(Ytr, Vtr, Wtr, Ztr, doW, doZ, reg = reg).detach()

            # Retraining hypers for calibration
            if retrain_hypers:
                self.train(Ycal, Acal, Vcal, 
                   reg = reg, niter = retrain_iters, learn_rate = retrain_lr, switch_grads_off = False)
            
            # Getting bootstrap indices
            bootstrap_inds1 = [torch.randint(0, ncal1, (ncal1,)) for _ in range(bootstrap_replications)]
            if two_datasets:
                bootstrap_inds0 = [torch.randint(0, ncal0, (ncal0,)) for _ in range(bootstrap_replications)]
            else:
                boostrap_inds0 = bootstrap_inds1
                
            # Looping over calibration parameter values and bootstrap datasets
            Post_levels = torch.zeros((len(nulist),len(doZ), len(levels)))
            for b in range(len(bootstrap_inds1)):
                    
                # Getting dataset
                Yb, Vb, Zb, = Ycal[bootstrap_inds1[b]], [Vcal[0][bootstrap_inds0[b]],Vcal[1][bootstrap_inds1[b]]], Zcal[bootstrap_inds0[b]]
                if W != []:
                    Wb = Wcal[bootstrap_inds1[b]]
                else:
                    Wb = []
             
                # Get posterior mean
                meanb = self.post_mean(Yb, Vb, Wb, Zb, doW, doZ, reg = reg).detach()

                # iterating over nulist, get post-var and indicator for is_inside_CI per level and do(Z)xdo(W)
                for k in range(len(nulist)):
                    varb = self.post_var(Yb, Vb, Wb, Zb, doW, doZ, reg = reg, latent = True, nu = nulist[k]).detach()
                    
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

 