import numpy as np
from scipy.special import erfc
from data_model import DataModelType


"""
------------------------------------------------------------------------------------------------------------------------
    Matrix Generation
------------------------------------------------------------------------------------------------------------------------
"""

def random_positive_definite_matrix(d, scaling = 50, variance = 2):
    """
    Returns a random positive definite matrix of size d x d
    """
    A = np.random.normal(0,variance,size=(d,d))
    return A.T @ A + scaling * np.eye(d)

def power_law_diagonal_matrix(d,alpha = 1.1):
    """
    Returns a diagonal matrix with a spectra that is power-law distributed
    """
    return np.diag([d/(k+1)**alpha for k in range(d)])

"""
------------------------------------------------------------------------------------------------------------------------
    Numerics
------------------------------------------------------------------------------------------------------------------------
"""

def stable_cosh(x):
    out = np.zeros_like(x)
    idx = x <= 0
    out[idx] = np.exp(x[idx]) / (1 + np.exp(2*x[idx]))
    idx = x > 0
    out[idx] = np.exp(-x[idx]) / (1 + np.exp(-2*x[idx]))
    return out

def stable_cosh_squared(x):
    out = np.zeros_like(x)
    idx = x <= 0
    out[idx] = np.exp(x[idx]) / (1 + np.exp(2*x[idx]) + 2*np.exp(x[idx]))
    idx = x > 0
    out[idx] = np.exp(-x[idx]) / (1 + np.exp(-2*x[idx]) + 2*np.exp(-x[idx]))
    return out

def sigmoid(x):
    out = np.zeros_like(x)
    idx = x <= 0
    out[idx] = np.exp(x[idx]) / (1 + np.exp(x[idx]))
    idx = x > 0
    out[idx] = 1 / (1 + np.exp(-x[idx]))
    return out

def log1pexp(x):
    out = np.zeros_like(x)
    idx0 = x <= -37
    out[idx0] = np.exp(x[idx0])
    idx1 = (x > -37) & (x <= -2)
    out[idx1] = np.log1p(np.exp(x[idx1]))
    idx2 = (x > -2) & (x <= 18)
    out[idx2] = np.log(1. + np.exp(x[idx2]))
    idx3 = (x > 18) & (x <= 33.3)
    out[idx3] = x[idx3] + np.exp(-x[idx3])
    idx4 = x > 33.3
    out[idx4] = x[idx4]
    return out

"""
------------------------------------------------------------------------------------------------------------------------
    Losses and Activations
------------------------------------------------------------------------------------------------------------------------
"""

def sigma_star(x):
    """
    Returns 0.5 * erfc(-x/sqrt(2))
    """
    return 0.5 * erfc(-x/np.sqrt(2))


def adversarial_loss(y,z, epsilon_term):
    return log1pexp(-y*z + epsilon_term)

def second_derivative_loss(y: float, z: float, epsilon_term: float) -> float:
    return y**2 * stable_cosh(0.5*y*z - 0.5*epsilon_term)**(2)

def gaussian(x : float, mean : float = 0, var : float = 1) -> float:
    '''
    Gaussian measure
    '''
    return np.exp(-.5 * (x-mean)**2 / var)/np.sqrt(2*np.pi*var)
    
"""
------------------------------------------------------------------------------------------------------------------------
    Helper Classes
------------------------------------------------------------------------------------------------------------------------
"""



class Task:
    def __init__(self, id, experiment_id, method, alpha, epsilon, lam, tau,d,ps, dp, data_model_type: DataModelType, compute_hessian: bool = False):
        self.id = id
        self.experiment_id = experiment_id
        self.method = method
        self.alpha = alpha
        self.epsilon = epsilon
        self.lam = lam
        self.tau = tau
        self.d = d
        self.gamma = 1
        self.result = None
        self.ps = ps
        self.dp = dp
        self.data_model_type: DataModelType = data_model_type
        self.compute_hessian: bool = compute_hessian

    def __str__(self):
        return f"Task {self.id} with method {self.method} and alpha={self.alpha}, epsilon={self.epsilon}, lambda={self.lam}, tau={self.tau}, d={self.d}, and data model {self.data_model_type.name}"
