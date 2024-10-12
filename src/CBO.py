import torch
import torch.nn as nn
import torch.optim
import sys
from src.GP_utils import GaussianProcess 

# CBO prior kernel class
class CBOPriorKernel:
    def __init__(self,main_kernel,prior_kernel):
        self.kernel = main_kernel
        self.prior_kernel = prior_kernel
        self.parameters = [self.prior_kernel.lengthscale,
                           self.prior_kernel.scale]

    def get_gram(self,X,Z):
        return self.kernel(X,Z)+self.prior_kernel.get_gram(X,Z)

class CausalKernel:
    def __init__(self, estimate_var_func, base_kernel=None, add_base_kernel=True):
        """
        CausalKernel that incorporates dynamic estimation of \hat{E}[Y|do(X=x)] and \hat Var [\hat E[Y|do(X=x)]].
        Optionally adds a base kernel (e.g., RBF) to the causal kernel.

        :param estimate_var_func: Function to estimate \hat{Var}[\hat{E}[Y|do(X=x)]] for any input X.
        :param base_kernel: A base kernel (e.g., RBF kernel) to add to the causal kernel.
        :param add_base_kernel: Boolean indicating whether to add the base kernel to the causal kernel.
        """
        super(CausalKernel, self).__init__()
        self.estimate_var_func = estimate_var_func
        self.base_kernel = base_kernel
        self.add_base_kernel = add_base_kernel
        self.parameters = [self.base_kernel.lengthscale,self.base_kernel.scale]

    def get_gram(self, X1, X2):
        """
        Computes the kernel matrix between X1 and X2, incorporating dynamically estimated \hat{E}[Y|do(X=x)] and \hat{Var}[\hat{E}[Y|do(X=x)]].
        Optionally adds a base kernel (e.g., RBF kernel) to the causal kernel.

        :param X1: First input tensor.
        :param X2: Second input tensor.
        :return: The kernel matrix incorporating causal estimates, and optionally a base kernel.
        """
        # Dynamically estimate \hat{E}[Y|do(X=x)] and \hat{Var}[\hat{E}[Y|do(X=x)]]
        sigma_X1 = torch.sqrt(self.estimate_var_func(X1))  # Standard deviation \sigma(x) from variance \hat{V}[E[Y|do(X)]]
        if X1.shape == X2.shape:
            if (X1-X2).abs().sum()==0:
                sigma_X2 = sigma_X1
        else:
            sigma_X2 = torch.sqrt(self.estimate_var_func(X2))  # Standard deviation \sigma(x')

        # Causal kernel component: \sigma(X1) \times \sigma(X2)^T
        causal_kernel = sigma_X1 @ sigma_X2.T

        # Optionally add the base kernel (e.g., RBF kernel)
        if self.add_base_kernel and self.base_kernel is not None:
            base_kernel_matrix = self.base_kernel.get_gram(X1, X2)
            return causal_kernel + base_kernel_matrix
        else:
            return causal_kernel

        
def expected_improvement(mu, sigma, y_best, xi=0.01, minimise = False):
    """Calculate the expected improvement."""
    sigma = sigma.clamp(min=1e-9)  # Numerical stability
    with torch.no_grad():
        if minimise:
            Z = -(mu - y_best - xi) / sigma
            ei = -(mu - y_best - xi) * torch.distributions.Normal(0, 1).cdf(Z) + sigma * torch.distributions.Normal(0, 1).log_prob(Z).exp()
        
        else:
            Z = (mu - y_best - xi) / sigma
            ei = (mu - y_best - xi) * torch.distributions.Normal(0, 1).cdf(Z) + sigma * torch.distributions.Normal(0, 1).log_prob(Z).exp()
        return ei
        
def causal_bayesian_optimization(X_train=None, y_train=None, kernel=None, mean = None, X_test=None, Y_test=None, n_iter=10, update_hyperparameters=True, update_interval=1, hyperparam_steps=100, lr=0.01, xi = 0.01, print_ = True, minimise = False, noise_init = -10.0, reg = 1e-3):
    """
    Causal Bayesian Optimization function with optional hyperparameter updating.

    :param X_train: Initial training inputs (can be None).
    :param y_train: Initial training outputs (can be None).
    :param kernel: Kernel to be used in the Gaussian Process model.
    :param mean: mean function to be used in the Gaussian Process model.
    :param X_test: Grid of test points (input values) for evaluation.
    :param Y_test: Precomputed grid of test outputs (true values).
    :param n_iter: Number of iterations for optimization.
    :param update_hyperparameters: Boolean indicating whether to update GP hyperparameters.
    :param update_interval: Number of iterations after which to update GP hyperparameters if `update_hyperparameters` is True.
    :param hyperparam_steps: Number of steps for hyperparameter optimization.
    :param lr: Learning rate for hyperparameter optimization.
    :return: Updated training inputs and outputs after optimization.
    """
    if X_train is None or y_train is None:
        X_train = torch.empty((0, X_test.shape[1]))  # Initialize empty tensor
        y_train = torch.empty((0, 1))  # Initialize empty tensor
    
    # Initialize Gaussian Process model with the initial data and kernel
    gp = GaussianProcess(X_train=X_train, y_train=y_train, kernel=kernel, noise_init = noise_init, mean = mean, nugget = reg)
    
    # Initialize the maximum observed value
    y_best = torch.max(y_train) if len(y_train) > 0 else 0
    x_best = 0.5
    true_x_best = X_test[torch.argmax(Y_test)]
    i = 0
    while i < n_iter and x_best != true_x_best:
        # Get the GP predictions for the test grid
        mu_s, cov_s = gp(X_test)
        sigma_s = torch.sqrt(torch.diag(cov_s).abs())
    
        # Calculate the Expected Improvement
        ei = expected_improvement(mu_s[:,0], sigma_s, y_best, xi = xi, minimise = minimise)
    
        # Find the next best point
        next_index = torch.argmax(ei)
        next_x = X_test[next_index]
        next_y = Y_test[next_index]
    
        # Update the training data with the new point
        X_train = torch.cat((X_train, next_x.unsqueeze(0)), dim=0)
        y_train = torch.cat((y_train, next_y.unsqueeze(0)), dim=0)
    
        # Update GP model with new data
        gp.X_train = X_train
        gp.y_train = y_train
    
        # Perform hyperparameter optimization if required
        if update_hyperparameters and (i + 1) % update_interval == 0:
            gp.optimize_hyperparameters(num_steps=hyperparam_steps, lr=lr, print_ = print_)
    
        # Update the best observed value
        if minimise:
            y_best = torch.min(y_train)
            x_best = X_train[torch.argmin(y_train)]
        else:
            y_best = torch.max(y_train)
            x_best = X_train[torch.argmax(y_train)]
            
        if print_:
            print(f"Iteration {i+1}: X = {x_best}, Y = {y_best}")
    
        i += 1
    if i < n_iter:
        X_train = torch.cat((X_train,x_best*torch.ones((n_iter - i,1))), dim = 0)
        y_train = torch.cat((y_train,y_best*torch.ones((n_iter - i,1))), dim = 0)

    return X_train, y_train