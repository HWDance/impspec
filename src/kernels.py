import torch

"""
Kernels and nuclear dominant kernels
"""

class Kernel:
    """
    Parameters
    ----------
    lengthscale: 
        torch.tensor, vector or matrix for scaling inputs X
    scale: 
        torch.tensor, scalar for kernel variance
    """

    def __init__(self,lengthscale=[],scale=[], hypers = []):
        if lengthscale == []:
            self.lengthscale = torch.ones(1,requires_grad = True)
        else:
            self.lengthscale = lengthscale
        if scale ==[]:
            self.scale = torch.ones(1).requires_grad_(True)
        else:
            self.scale = scale
        if hypers == []:
            self.hypers = torch.ones(1).requires_grad_(True)
        else:
            self.hypers = hypers
        
    def get_gram(self, X : torch.tensor, Z : torch.tensor):
        # Computes K_XZ
        raise NotImplementedError

class GaussianKernel(Kernel):
        
    def get_gram(self,X,Z):
        K_xz = torch.exp(-0.5*torch.cdist(X/self.lengthscale, Z/self.lengthscale, p=2.0)**2)            
        return K_xz*self.scale**2
        
class ExponentialKernel(Kernel):
        
    def get_gram(self,X,Z):
        K_xz = torch.exp(-0.5*torch.cdist(X/self.lengthscale, Z/self.lengthscale, p=2.0)**1)            
        return K_xz*self.scale**2

class GammaExponentialKernel(Kernel):
    def get_gram(self,X,Z):
        K_xz = torch.exp(-0.5*torch.cdist(X/self.lengthscale, Z/self.lengthscale, p=2.0)**(1+(1/(1+self.hypers**2))))            
        return K_xz*self.scale**2

class MultivariateGaussianKernel(Kernel):
    
    def get_gram(self,X,Z):
        K_xz = torch.exp(-0.5*torch.cdist(X @ self.lengthscale, Z @ self.lengthscale, p=2.0)**2)
        return K_xz*self.scale**2

class LinearKernel(Kernel):

    def get_gram(self,X,Z):
        return X*self.lengthscale @ (Z*self.lengthscale).T

# Replaces get_nuclear_kernel (for Gaussian kernel wih N(0,sigma)) measure
class NuclearGaussianKernel(Kernel):

    def __init__(self,lengthscale, scale, sigma):
        super().__init__(lengthscale, scale)
        self.sigma = sigma

    def get_gram(self,X,Z):
        d = X.size()[1]
        M = torch.diag(1/self.lengthscale**2)
        A = 2*M+torch.eye(d)/self.sigma**2
        B_inv = 1/(1/(2*M)+torch.eye(d)*self.sigma**2)
        normaliser = torch.diag(A).prod()**-0.5
        exponent1 = -1/2*torch.cdist(X/(self.lengthscale*2**0.5), Z/(self.lengthscale*2**0.5), p=2.0)**2
        exponent2 = -1/8*torch.cdist(X*torch.diag(B_inv)**0.5, -Z*torch.diag(B_inv)**0.5, p=2.0)**2
        
        return self.scale**4*normaliser*torch.exp(exponent1+exponent2)*self.sigma**-d

    def get_gram_base(self,X,Z):
        K_xz = torch.exp(-0.5*torch.cdist(X/self.lengthscale, Z/self.lengthscale, p=2.0)**2)            
        return K_xz*self.scale**2
        
class NuclearKernel:
    """
    Parameters
    ----------
    base_kernel:
        Class, an instantiated kernel
    dist:
        torch.distribution, integrating distribution
    """

    def __init__(self, base_kernel, dist, samples):
        self.base_kernel = base_kernel
        self.dist = dist
        self.samples = samples

    def get_gram_approx(self, X: torch.tensor, Z: torch.tensor, rsample = False):
        """Returns approximated gram matrix of nuclear dominant kernel"""
        if rsample:
            U = self.dist.rsample((self.samples,))
        else:
            U = self.dist.sample((self.samples,))
        K_xu = self.base_kernel.get_gram(X,U)
        K_uz = self.base_kernel.get_gram(U,Z)
        return K_xu @ K_uz/self.samples

    def get_gram_gaussian(self, X: torch.tensor, Z: torch.tensor):
        """Returns exact gram matrix of Gaussian nuclear dominant kernel kernel"""
        assert(self.dist.loc.abs().max() == 0)    
        d = X.size()[1]
        M = torch.diag(1/self.base_kernel.lengthscale**2)
        A = 2*M+torch.eye(d)/self.dist.scale**2
        B_inv = 1/(1/(2*M)+torch.eye(d)*self.dist.scale**2)
        normaliser = torch.diag(A).prod()**-0.5
        exponent1 = -1/2*torch.cdist(X/(self.base_kernel.lengthscale*2**0.5), Z/(self.base_kernel.lengthscale*2**0.5), p=2.0)**2
        exponent2 = -1/8*torch.cdist(X*torch.diag(B_inv)**0.5, -Z*torch.diag(B_inv)**0.5, p=2.0)**2
        
        return self.base_kernel.scale**4*normaliser*torch.exp(exponent1+exponent2)*self.dist.scale**-d
        
    def get_gram_base(self, X: torch.tensor, Z: torch.tensor):
        """Returns gram matrix of base kernel"""
        return self.base_kernel.get_gram(X,Z)

class ProductKernel:
    """
    Parameters
    ----------
    kernels:
        list of Kernel() objects
    """
    def __init__(self, kernels):
        self.kernels = kernels

    def get_gram(self, X, Z):
        """ X, Z as lists of length len(kernels), 
            each element as N x P_i """
        assert (len(X) == len(Z))

        K = torch.ones((len(X[0]),len(Z[0])))

        for i in range(len(X)):
            K *= self.kernels[i].get_gram(X[i],Z[i])

        return K
            



