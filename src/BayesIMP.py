import torch

class BayesIMP:
    """
    BayesIMP method for estimating posterior moments of average causal effects
    """

    def __init__(self,kernel_A, kernel_V, kernel_Z, noise_Y, noise_feat):
        self.kernel_A = kernel_A
        self.kernel_V = kernel_V
        self.kernel_Z = kernel_Z
        self.noise_Y = noise_Y
        self.noise_feat = noise_feat

    """Compute E[E[Y|do(A)]] in A -> V -> Y """
    def post_mean(self, Y, A, V, doA, reg = 1e-4):
        
        n = len(Y)
        
        # getting kernel matrices
        R_vv,K_aa,k_atest = (self.kernel_V.get_gram(V,V),
                             self.kernel_A.get_gram(A,A),
                             self.kernel_A.get_gram(doA, A))
        R_v = R_vv+(self.noise_Y+reg)*torch.eye(n)
        K_a = K_aa+(self.noise_feat+reg)*torch.eye(n)

        # Getting components
        A_a = torch.linalg.solve(K_a,k_atest.T).T
        alpha_y = torch.linalg.solve(R_v,Y)
        
        return  A_a @ R_vv @ alpha_y
        
    """Compute Var[E[Y|do(A)]] in A -> V -> Y """
    def post_var(self, Y, A, V, doA, reg = 1e-4, latent = True):
        
        n = len(Y)
        
        # getting kernel matrices
        R_vv,K_vv,K_aa,k_atest = (self.kernel_V.get_gram(V,V),
                                 self.kernel_V.get_gram_base(V,V),
                                 self.kernel_A.get_gram(A,A),
                                 self.kernel_A.get_gram(doA, A))
        R_v = R_vv+(self.noise_Y+reg)*torch.eye(n)
        K_v = K_vv+(self.noise_Y+reg)*torch.eye(n)
        K_a = K_aa+(self.noise_feat+reg)*torch.eye(n)
        R_vv_bar = R_vv - R_vv @ torch.linalg.solve(R_v,R_vv)+ (not latent)*self.noise_Y*torch.eye(n)
        kpost_atest_approx = (k_atest @ k_atest.T - k_atest @ torch.linalg.solve(K_a,k_atest.T))+ (not latent)*self.noise_feat*torch.eye(len(doA))
        
        # computing matrix vector products
        alpha_a = torch.linalg.solve(K_a,k_atest.T)
        alpha_y = torch.linalg.solve(K_v,Y)
        KinvR = torch.linalg.solve(K_vv+torch.eye(n)*reg,R_vv)
        
        V1 = alpha_a.T @ R_vv_bar @ alpha_a
        V2 = alpha_y.T @ R_vv @ KinvR @ KinvR @ alpha_y * kpost_atest_approx
        V3 = torch.trace(torch.linalg.solve(K_vv+torch.eye(n)*reg, R_vv_bar @ KinvR)) * kpost_atest_approx
        
        return V1+V2+V3