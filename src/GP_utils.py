import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal

class GaussianProcess(nn.Module):
    def __init__(self, X_train=None, y_train=None, kernel=None, noise_init=1e-5, nugget=1e-6, mean=None):
        super(GaussianProcess, self).__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.kernel = kernel
        
        # Initialize noise hyperparameter
        self.noise = nn.Parameter(torch.tensor(noise_init))

        # Nugget regularization term
        self.nugget = nugget

        # Mean function
        self.mean_func = mean if mean is not None else (lambda X: torch.zeros(X.shape[0], 1))

    def forward(self, X_test):
        if self.X_train is None or self.y_train is None or self.X_train.numel() == 0 or self.y_train.numel() == 0:
            # If there's no training data, return mean m(X_test) and large covariance
            mu_s = self.mean_func(X_test)
            cov_s = self.kernel.get_gram(X_test, X_test) + (self.noise.exp() + self.nugget) * torch.eye(len(X_test))
            return mu_s, cov_s
        else:
            # Compute the Gram matrices for training data and between training and test data
            K = self.kernel.get_gram(self.X_train, self.X_train) + (self.noise.exp() + self.nugget) * torch.eye(len(self.X_train))
            L = torch.linalg.cholesky(K)  # Use safe Cholesky decomposition

            # Calculate the centered y_train
            y_train_centered = self.y_train - self.mean_func(self.X_train)

            # Solve for alpha = K_inv @ (y_train - m(X_train)) using Cholesky solve
            alpha = torch.cholesky_solve(y_train_centered, L)

            # Compute the mean at test points
            K_s = self.kernel.get_gram(self.X_train, X_test)
            mu_s = self.mean_func(X_test) + K_s.T @ alpha

            # Compute the covariance at test points
            K_ss = self.kernel.get_gram(X_test, X_test) + (self.noise.exp() + self.nugget) * torch.eye(len(X_test))
            v = torch.cholesky_solve(K_s, L)
            cov_s = K_ss - K_s.T @ v

            # Check for NaNs and handle them
            if torch.isnan(cov_s).any():
                print("NaNs detected in covariance matrix")
                cov_s = torch.nan_to_num(cov_s, nan=1e-9)

            return mu_s, cov_s

    def optimize_hyperparameters(self, num_steps=1, lr=0.01, print_ = True):
        """
        Optimize the GP hyperparameters using gradient descent.

        :param num_steps: Number of optimization steps.
        :param lr: Learning rate for the optimizer.
        """
        # Hyperparameters to be optimized (only noise in GP, others in kernel)
        params = [self.noise] + self.kernel.parameters

        optimizer = torch.optim.Adam(params, lr=lr)

        for step in range(num_steps):
            optimizer.zero_grad()

            # Compute the negative log marginal likelihood (NLML)
            K = self.kernel.get_gram(self.X_train, self.X_train) + self.noise.exp() * torch.eye(len(self.X_train)) + self.nugget * torch.eye(len(self.X_train))
            L = torch.linalg.cholesky(K)
            alpha = torch.cholesky_solve(self.y_train, L)

            log_likelihood = -0.5 * self.y_train.T @ alpha
            log_likelihood -= torch.sum(torch.log(torch.diag(L)))
            
            loss = -log_likelihood  # We minimize the negative log likelihood

            # Backpropagate the gradient
            loss.backward()

            # Perform a gradient step
            optimizer.step()
            
            if print_:
                if step % 10 == 0:
                    print(f"Step {step}/{num_steps}, Loss: {loss.item():.4f}")

        if print_:
            print("Optimization completed.")

def GPML(Y, X, kernel_X, noise_Y,reg = 1e-4, force_PD = False):
    n = len(X)
    K_xx = kernel_X.get_gram(X,X)
    if force_PD:
        K_xx = (K_xx+K_xx.T)/2
    L = torch.linalg.cholesky((K_xx + (noise_Y+reg)*torch.eye(n)).double())
    alpha = torch.cholesky_solve(Y.reshape(n,1).double(), L).float()

    log_likelihood = -0.5 * Y.reshape(n,1).T @ alpha
    log_likelihood -= torch.sum(torch.log(torch.diag(L)))
    return log_likelihood    
    #return MultivariateNormal(torch.zeros(n),K_xx+(noise_Y+reg)*torch.eye(n)).log_prob(Y).sum()

def GPfeatureML(Y, X, kernel_Y, kernel_X, noise_feat, reg = 1e-4):
    n = len(Y)
    R_yy = kernel_Y.get_gram(Y, Y)
    K_yy = kernel_Y.get_gram_base(Y, Y)
    K_xx = kernel_X.get_gram(X, X)
    K_x = K_xx + (noise_feat+reg)*torch.eye(n)
    R_y = R_yy+reg*torch.eye(n)
    ml =  -(n/2*(torch.logdet(K_x)+torch.logdet(R_y))
             +1/2*(torch.trace(torch.linalg.solve(K_x, K_yy @ torch.linalg.solve(R_y, K_yy))))
            )
    return ml

def GPmercerML(Y, X, kernel_Y, kernel_X, noise_feat,reg = 1e-4, force_PD = False):
    n = len(Y)
    K_yy = kernel_Y.get_gram_base(Y, Y)
    K_xx = kernel_X.get_gram(X, X)
    K_x = K_xx + (noise_feat+reg)*torch.eye(n)
    if force_PD:
        K_x = 0.5*(K_x + K_x.T)
    ml =  -(K_yy[0,0]*1/2*torch.logdet(K_x)
             +1/2*torch.trace(torch.linalg.solve(K_x, K_yy))
            )
    return ml

def GPpostmean(Y, X, Xtest, kernel, noise, reg = 1e-4):
    n,m = len(X),len(Xtest)
    K_xx = kernel.get_gram(X,X)
    K_xxtest = kernel.get_gram(X,Xtest)
    K_x = K_xx + (noise+reg)*torch.eye(n)
    return  K_xxtest.T @ torch.linalg.solve(K_x, Y)

def GPpostvar(X, Xtest, kernel, noise, reg=1e-4, latent=False, diag=False):
    n, m = len(X), len(Xtest)
    
    # Compute the kernel matrices
    K_xx = kernel.get_gram(X, X)  # (n, n)
    K_xxtest = kernel.get_gram(X, Xtest)  # (n, m)
    
    # Add noise and regularization to the training kernel matrix
    K_x = K_xx + (noise + reg) * torch.eye(n)
    
    if diag:
        # Compute only the diagonal of K(Xtest, Xtest)
        K_xtestxtest = kernel.get_gram(Xtest[:, None], Xtest[:, None]).squeeze()
        B = (torch.linalg.solve(K_x, K_xxtest) * K_xxtest).sum(dim=0)  # (m,)
    else:
        # Compute the full K(Xtest, Xtest) matrix
        K_xtestxtest = kernel.get_gram(Xtest, Xtest)  # (m, m)
        B = K_xxtest.T @ torch.linalg.solve(K_x, K_xxtest)  # (m, m)
        
    posterior_variance = K_xtestxtest - B
    
    if not latent:
        posterior_variance += noise * torch.eye(m) if not diag else noise
    
    return posterior_variance
    
def GP_cal(Y, mean, var, levels):
    """
    Y:
        N x 1 array of outputs
    mean:
        N x 1 array of means
    var:
        N x 1 array of variances
    levels:
        P x 1 array of confidence levels
    """
    
    z_quantiles = Normal(0, 1).icdf(1-(1-levels)/2)
    posterior_fraction = ((Y - mean).abs() <= var**0.5 @ z_quantiles.T).float().mean(0)

    return posterior_fraction

    