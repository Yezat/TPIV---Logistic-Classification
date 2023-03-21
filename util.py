import numpy as np
from scipy.special import erfc

def error(y, yhat):
    return 0.25*np.mean((y-yhat)**2)

def sigma_star(x):
    """
    Returns 0.5 * erfc(-x/sqrt(2))
    """
    return 0.5 * erfc(-x/np.sqrt(2))

def logistic_function(x):
    """
    Returns the logistic function of x
    """
    return 1/(1+np.exp(-x))

def generalization_error(rho_w_star,m,Q):
    """
    Returns the generalization error in terms of the overlaps
    """
    return np.arccos(m / np.sqrt( rho_w_star * Q ) )/np.pi


